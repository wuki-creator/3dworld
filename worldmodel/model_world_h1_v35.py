# -*- coding: utf-8 -*-
"""MagWorld v35 = v25 磁流冠军 + 极性调控规则 + 因果自回归传播.

用户新方案（2026-10-07）：
  「扰动因子发出磁场，基因如果是同极，则下调，如果是异极，则上调。
    这个调控构建具有因果关系的自回归控制。」

实现（v25 本体全部保留，只动响应规则与矩的传播符号）：

  1. 基因极性：pole_i = tanh(raw_pole_i) ∈ (-1, 1)，每基因可学习磁极。
     扰动源（被敲基因）以它自己的极性发射带符号场：
         S_i^0 = pole_src · φ_i^0        （φ 为 v25 的源标量场）

  2. 同极下调 / 异极上调：场每步沿 kNN 图传播时都带发射者的极性
     （signed moment = pole ⊙ moment），基因 i 收到的场 S_i 携带源极
     性符号。响应规则：
         resp_i = tanh( pole_i · S_i / τ )   # 同极→正，异极→负
         dx_i  = a_i · gate_i · ( tanh(r_i) − w_resp · resp_i )
     同极（pole_i·S_i>0）→ −w_resp·resp < 0 → 下调；异极 → 上调。✓

  3. 因果自回归控制：n_layers 步弛豫 = 信息沿图一跳一跳外扩的因果锥
     （t 步只影响 t 跳邻域，无瞬时全对全）。且矩的演化带符号：
         moment ← moment + relax · dx     （dx 带符号）
     被上调的磁体发射增强、被下调的减弱甚至反号 → 下一代场由上一代
     响应因果决定 —— 这就是自回归控制链，而不是一步到位的回归。

  4. 零初始化门 w_resp = tanh(raw_resp) = 0 → 初始函数严格 ≡ v25；
     w_resp 先动（它对 pole 的梯度非零），pole 随后被带着学。
     τ = softplus(raw_tau) 可学习温度，控制同极/异极判别的软硬。

纪律：不改先验编码器、不改三条静态注入通道、不动 DGIDB hook；
除 pole 响应项外全部与 v25 一致。先验图参数照常传入（v25 逻辑）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import _slog
from model_world_h1_v25 import WorldModelH1V25


class WorldModelH1V35(WorldModelH1V25):
    def __init__(
        self,
        *args,
        pole_init_std: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        G = self.n_genes
        # ---- v35 极性规则参数（不依赖 prior graph，必定创建）----
        self.raw_pole = nn.Parameter(
            torch.randn(G) * float(pole_init_std))
        self.raw_tau = nn.Parameter(torch.tensor(1.0))
        self.raw_resp = nn.Parameter(torch.zeros(1))  # w_resp=tanh(0)=0 ≡ v25

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
        smoment = pole[None, :] * moment                            # 带符号矩 (B,G)

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

        # ---- v35 带符号极性场：源以自己的极性发射 ----
        S = pole[indices][:, None] * phi                            # (B,G)

        h = torch.zeros(B, G, device=device)
        relax = F.softplus(self.raw_relax)
        h_max = F.softplus(self.raw_h_max)[None, :]
        tau = F.softplus(self.raw_tau)
        w_resp = torch.tanh(self.raw_resp)                          # 零初始化
        for _ in range(self.n_layers):
            m_nb = moment[:, nb.reshape(-1)].reshape(B, G, nb.shape[1])
            sm_nb = smoment[:, nb.reshape(-1)].reshape(B, G, nb.shape[1])
            phi = phi + (k_nb[None] * m_nb).sum(-1)
            vec = vec + ((k_nb[None] * m_nb).unsqueeze(-1) * uvec[None]).sum(2)
            # 极性场沿图传播：每跳带发射者的符号（因果链的载体）
            S = S + (k_nb[None] * sm_nb).sum(-1)
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

            # ---- v35 极性规则：同极下调 / 异极上调 ----
            S_n = S / S.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            resp = torch.tanh(pole[None, :] * S_n / tau)            # 同极→+
            # v25 原规则 + 零初始化门控的极性修正
            dx = a * gate * (torch.tanh(r) - w_resp * resp)
            h = h + dx
            # 自回归：带符号响应回写矩 → 下一代发射符号因果取决于本代响应
            moment = moment + relax * dx
            smoment = pole[None, :] * moment

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
        diag["w_resp"] = round(float(torch.tanh(self.raw_resp)), 4)
        diag["tau"] = round(float(F.softplus(self.raw_tau)), 4)
        return diag


WorldModel = WorldModelH1V35
