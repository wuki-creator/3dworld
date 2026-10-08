# -*- coding: utf-8 -*-
"""MagWorld v36 = v25 磁流冠军 + 向量因果 Transformer（因果 mask + 极性偏置）.

用户方案（2026-10-07）：「向量计算表达因果，同时用 transformer＋mask 架构」。

因果序：基因无天然序列，磁场图里有 —— 到扰动源的几何距离 d_src 就是
物理因果锥的序（场先近后远）。mask 规则：基因 i 只注意严格比它靠近源
的基因（d_j < d_i），完全向量化，零预计算。

结构（每层 = 自回归的一步，共 n_layers 步）：
    v_i^0 = W_v s_i                                    # s: v25 现成 per-sample 基因态
    第 t 层（对 kNN 邻居 j ∈ N_k(i)）：
        score_ij = (Wq v_i)·(Wk v_j)/√d_g
                 − β · pole_i · pole_j               # 极性偏置：异极相吸（磁语义进注意力）
                 − ∞ · [d_j ≥ d_i]                   # 因果 mask
        v_i ← LayerNorm(v_i + W_o · softmax_j(score) V_j)
        v_i ← LayerNorm(v_i + FFN(v_i))
    T 层后：dx_tf = tanh(w_head · v^T)，w_head 零初始化。

因果链成立的关键：第 t 层里距源 t 跳的基因，读到的是第 t−1 层更新过的
t−1 跳基因状态 —— 响应沿距离序逐跳生成（向量值自回归），全程可并行
（WaveNet 式因果卷积的图版本）。

纪律：计算量 ≡ v33 GAT 量级（k=50 邻居、单头 d_g=64、(B,G,k) 中间张量）；
先验编码器与三条静态注入通道原封不动；w_head 零初始化 → 初始函数 ≡ v25。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import _slog
from model_world_h1_v25 import WorldModelH1V25


class WorldModelH1V36(WorldModelH1V25):
    def __init__(
        self,
        *args,
        d_gat: int = 64,
        pole_init_std: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        G = self.n_genes
        dm = self.d_dir          # s = state_gene + state_ctx 的维度（v18 为 d_dir）
        dg = int(d_gat)

        # ---- 基因极性（注意力偏置用）----
        self.raw_pole = nn.Parameter(torch.randn(G) * float(pole_init_std))
        self.raw_beta = nn.Parameter(torch.tensor(1.0))  # 异极相吸强度

        # ---- 因果 transformer 参数 ----
        self.tf_q = nn.Linear(dm, dg, bias=False)
        self.tf_k = nn.Linear(dm, dg, bias=False)
        self.tf_v = nn.Linear(dm, dg, bias=False)
        self.tf_o = nn.Linear(dg, dm, bias=False)
        self.tf_norm1 = nn.LayerNorm(dm)
        self.tf_ffn1 = nn.Linear(dm, 2 * dm)
        self.tf_ffn2 = nn.Linear(2 * dm, dm)
        self.tf_norm2 = nn.LayerNorm(dm)
        # 零初始化读出头 → 初始函数严格 ≡ v25
        self.tf_head = nn.Linear(dm, 1, bias=False)
        nn.init.zeros_(self.tf_head.weight)

    # ------------------------------------------------------------ forward
    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        if self.n_relations == 0:
            return super().predict_delta(x_ctrl, pert_idx)
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        B = x_ctrl.shape[0]
        e = self.gene_emb.weight
        G = e.shape[0]

        u = self._prior_encode(e)                                   # (G, d_z)
        charge_prior = self.prior_charge(u).squeeze(-1)             # (G,)
        amp_prior = self.prior_amp(u).squeeze(-1)                   # (G,)
        src_prior = 1.0 + torch.tanh(
            self.prior_src(u[indices]).squeeze(-1))                 # (B,)

        pole = torch.tanh(self.raw_pole)                            # (G,)
        beta = self.raw_beta

        pos = F.normalize(self.pole(e), dim=-1)
        charge = F.softplus(
            self.charge(e).squeeze(-1) + self.charge_bias + charge_prior
        )

        nb = self._knn_edges(pos.detach())                          # (G, k)
        p_nb = pos[nb]
        diff = pos[:, None, :] - p_nb
        r_nb = diff.norm(dim=-1).clamp_min(1e-4)
        k_nb = self._kernel(r_nb)
        k_nb = k_nb / (k_nb.sum(-1, keepdim=True) + 1e-6)
        uvec = diff / r_nb.unsqueeze(-1)

        ctx_in = F.layer_norm(x_ctrl, (G,)) if self.normalize_context else x_ctrl
        z = self.context_encoder(ctx_in)
        s = self.state_gene(e)[None, :, :] + self.state_ctx(z)[:, None, :]

        expr = torch.log1p(x_ctrl.clamp_min(0.0))
        moment = charge[None, :] * (
            1.0 + torch.tanh(self.moment_expr_w * expr + self.moment_expr_b)
        )

        kappa = F.softplus(self.raw_source_kappa)
        p_src = pos[indices]
        src_diff = pos[None, :, :] - p_src[:, None, :]
        d_src = src_diff.norm(dim=-1).clamp_min(1e-4)               # (B,G) 因果序
        k_src = self._kernel(d_src)
        k_src = k_src / (k_src.mean(dim=1, keepdim=True) + 1e-6)
        u_src = src_diff / d_src.unsqueeze(-1)

        # ---- v18 DGIDB hooks + static prior src channel ----
        f = getattr(self, "dgidb_feat", None)
        if f is not None:
            f_src = f[indices]
            src_gain = (1.0 + torch.tanh(self.src_proj(f_src)).squeeze(-1)) * src_prior
            amp_bias = self.amp_dg(f).squeeze(-1) + amp_prior
        else:
            src_gain = src_prior
            amp_bias = amp_prior

        phi = kappa * charge[indices][:, None] * k_src * src_gain[:, None]
        vec = phi[:, :, None] * u_src

        # ---- v36 因果 transformer：v 在 n_layers 步里逐跳更新 ----
        # 注意：transformer 状态用独立变量 s_tf，不改 s ——
        # 磁流弛豫用的仍是 v25 原状态，保证零初始化头下初始函数 ≡ v25
        s_tf = s
        v = self.tf_v(s_tf)                                         # (B,G,d_g)
        # 邻居的因果序与极性（静态部分预取）
        d_nb = d_src[:, nb.reshape(-1)].reshape(B, G, nb.shape[1])  # (B,G,k)
        pole_nb = pole[nb]                                          # (G,k)
        # 第 t 层允许注意「t−1 跳已更新」的基因：严格更近
        for _ in range(self.n_layers):
            q = self.tf_q(s_tf)                                     # (B,G,d_g)
            kk = self.tf_k(s_tf)                                    # (B,G,d_g)
            k_gather = kk[:, nb.reshape(-1)].reshape(B, G, nb.shape[1], -1)
            v_gather = v[:, nb.reshape(-1)].reshape(B, G, nb.shape[1], v.shape[-1])
            score = (q[:, :, None, :] * k_gather).sum(-1) / (q.shape[-1] ** 0.5)
            # 极性偏置：异极相吸（pole_i·pole_j < 0 → 加分）
            score = score - beta * pole[None, :, None] * pole_nb[None]
            # 因果 mask：只允许严格更靠近源的邻居
            causal = (d_nb < d_src[:, :, None] - 1e-6)
            score = score.masked_fill(~causal, float("-inf"))
            attn = torch.softmax(score, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            msg = (attn.unsqueeze(-1) * v_gather).sum(2)            # (B,G,d_g)
            s_tf = self.tf_norm1(s_tf + self.tf_o(msg))
            s_tf = self.tf_norm2(s_tf + self.tf_ffn2(F.gelu(self.tf_ffn1(s_tf))))
        dx_tf = torch.tanh(self.tf_head(s_tf).squeeze(-1))          # (B,G)

        h = torch.zeros(B, G, device=device)
        relax = F.softplus(self.raw_relax)
        h_max = F.softplus(self.raw_h_max)[None, :]
        for _ in range(self.n_layers):
            m_nb = moment[:, nb.reshape(-1)].reshape(B, G, nb.shape[1])
            phi = phi + (k_nb[None] * m_nb).sum(-1)
            vec = vec + ((k_nb[None] * m_nb).unsqueeze(-1) * uvec[None]).sum(2)
            v_norm = vec.norm(dim=-1).clamp_min(1e-6)
            if self.use_langevin:
                scale = h_max * self._langevin(v_norm / h_max) / v_norm
            else:
                scale = 1.0 / v_norm
            vec_s = vec * scale.unsqueeze(-1)

            h_ext = v_norm.mean(1, keepdim=True)
            pre_a = (
                self.amp_gene(s).squeeze(-1)
                + self.amp_phi(_slog(phi).unsqueeze(-1)).squeeze(-1)
                + self.amp_v(_slog(v_norm).unsqueeze(-1)).squeeze(-1)
                + self.amp_ext(h_ext)
                + self.amp_bias
                + amp_bias[None, :]
            )
            a = torch.sigmoid(pre_a)

            d0 = F.normalize(self.dir_proj(s), dim=-1)
            vproj = vec_s - (vec_s * d0).sum(-1, keepdim=True) * d0
            eta = torch.sigmoid(self.raw_eta) * 2.0
            d = F.normalize(d0 + eta * vproj, dim=-1)
            gate = torch.sigmoid(self.gate_proj(d).squeeze(-1) + self.gate_bias)

            phi_n = _slog(phi)
            phi_n = phi_n / phi_n.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            r = self.response_right(phi_n) @ self.response_left.weight

            dx = a * gate * (torch.tanh(r) + dx_tf)
            h = h + dx
            moment = moment + relax * dx

        ctx_rec = F.normalize(self.context_receiver(e), dim=-1)
        ctx_gate = self.context_strength * torch.tanh(
            z @ ctx_rec.T / (self.d_z ** 0.5)
        )
        shared = self.effective_shared_bias_scale()
        base = shared * self.gene_bias[None, :] * (1.0 + ctx_gate)
        raw = base + h

        self_effect = -F.softplus(self.raw_self_effect)
        raw = raw.scatter_add(
            1, indices[:, None], self_effect.expand(len(indices), 1)
        )
        return self.max_delta * torch.tanh(raw / self.max_delta)

    @torch.no_grad()
    def magnetic_diagnostics(self, pert_idx) -> dict:
        diag = super().magnetic_diagnostics(pert_idx)
        pole = torch.tanh(self.raw_pole)
        diag["pole_pos_frac"] = round(float((pole > 0.3).float().mean()), 4)
        diag["pole_neg_frac"] = round(float((pole < -0.3).float().mean()), 4)
        diag["beta_attract"] = round(float(self.raw_beta), 4)
        diag["tf_head_w"] = round(float(self.tf_head.weight.abs().sum()), 4)
        return diag


WorldModel = WorldModelH1V36
