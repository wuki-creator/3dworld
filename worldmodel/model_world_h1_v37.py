# -*- coding: utf-8 -*-
"""MagWorld v37 = v25 磁流冠军 + kq 作用力因果链（无 kNN，t 步自回归）.

用户方案（2026-10-07 22:55，23:00 修订：t 步循环也砍掉）：
  「不需要 knn，t 步作为因果链。用向量乘以 kq 矩阵作为磁极的因果，
    kq 矩阵为扰动因子和基因作用力，乘以神经网络向量后，
    负的为下调，正的为上调。」→ 最终版：无 kNN、无 t 步循环，单次 kq 力。

实现（v25 本体保留，新增一条并行的单次 kq 作用力通道）：

  扰动因子 token：src = W_src([e_src, z])        （源基因嵌入 + 样本上下文）
  每基因状态：    s_i = state_gene(e_i) + state_ctx(z)     （v25 现成）

  单次计算（v37 final：向量力 + 双投影 + 基因响应轴软归一化）：
      q_i = Wq s_i                    # 基因查询/接收轴       (B,G,d)
      k   = Wk src                    # 扰动因子场向量        (B,d)
      E_i = k ⊙ q_i                   # 向量力：场在基因接收方向上的逐元素调制
      v_i = Wv s_i                    # 神经网络值向量        (B,G,d)
      u_i = E_i ⊙ v_i                 # 诱导响应向量（向量场 × 值）
      π̂_i = π_i / (1 + ‖π_i‖)       # 基因响应轴（软归一化：方向唯一、强度≤1，
                                      #   零点梯度=恒等；强度解耦给幅度 a_i）
      dx_tf = π̂_i · u_i               # 投影：响应落在该基因效应轴上
                                      # 负 = 下调，正 = 上调

  并入 v25 响应：dx = a · gate · (tanh(r) + dx_tf)
  （w_out 零初始化 → 初始函数严格 ≡ v25 冠军）

成本：三次 (B,G,d) 矩阵乘 + 一次逐元素乘，全场最便宜的新通道。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import _slog
from model_world_h1_v25 import WorldModelH1V25


class WorldModelH1V37(WorldModelH1V25):
    def __init__(
        self,
        *args,
        d_gat: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        dm = self.d_dir          # s / k 所在空间维（v18 为 d_dir）
        dz = self.d_z
        dg = int(d_gat)

        # ---- 扰动因子 token 编码器（输出到 d_dir，与基因查询同空间）----
        self.src_tf = nn.Linear(self.d_model + dz, dm)
        # ---- kq 作用力 + 值向量 ----
        self.kq_q = nn.Linear(dm, dg, bias=False)
        self.kq_k = nn.Linear(dm, dg, bias=False)
        self.kq_v = nn.Linear(dm, dg, bias=False)
        # 每基因响应/磁极轴 π_i：零初始化（初始 ≡ v25，∂dx/∂π = u ≠ 0 梯度即通），
        # 使用时软归一化 π̂ = π/(1+‖π‖) → 只表方向，强度解耦给幅度 a_i
        self.gene_axis = nn.Parameter(torch.zeros(self.n_genes, dg))

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

        pos = F.normalize(self.pole(e), dim=-1)
        charge = F.softplus(
            self.charge(e).squeeze(-1) + self.charge_bias + charge_prior
        )

        nb = self._knn_edges(pos.detach())
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
        d_src = src_diff.norm(dim=-1).clamp_min(1e-4)
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

        # ---- v37 kq 作用力（单次，无 kNN、无 t 步循环）----
        src_vec = torch.tanh(self.src_tf(
            torch.cat([e[indices], z], dim=-1)))                    # (B,dm)
        q = self.kq_q(s)                                            # (B,G,dg)
        k = self.kq_k(src_vec)                                      # (B,dg) 场向量
        E = k[:, None, :] * q                                       # (B,G,dg) 向量力
        v = self.kq_v(s)                                            # (B,G,dg)
        u_vec = E * v                                               # 诱导响应向量
        axis = self.gene_axis / (
            1.0 + self.gene_axis.norm(dim=-1, keepdim=True))        # 软归一化
        dx_tf = (u_vec * axis[None]).sum(-1)                        # 投影：负↓ 正↑

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
        ga = self.gene_axis
        ghat = ga / (1.0 + ga.norm(dim=-1, keepdim=True))
        diag["axis_nonzero_frac"] = round(
            float((ga.norm(dim=-1) > 1e-4).float().mean()), 4)
        diag["axis_hat_len_mean"] = round(float(ghat.norm(dim=-1).mean()), 4)
        return diag


WorldModel = WorldModelH1V37
