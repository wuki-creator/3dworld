# -*- coding: utf-8 -*-
"""MagWorld v34 = v25 磁流冠军 + 掩码图先验编码器（磁流本体完全不动）.

相对 v25 的唯一改动：

  1. 先验编码器：v25 的「关系加权和」→「掩码图注意力」
       score_ij = (Wq u_i)·(Wk u_j)/√d   （仅先验边，双向化）
       α_ij     = σ(score_ij)·σ(log w_ij)
       u        = prior_norm(prior_self(e) + W_out·GAT(e))   # → d_z
     下游仍是 v25 的三条零初始化静态注入通道（charge/amp/src）。

  与 v33 的区别：v33 在弛豫循环里另加了第 4 条「磁流掩码通道」
  （α 加权的 moment 沿先验边传播），v34 把它整条删掉 —— 磁流本体
  与 v25 逐行一致。用于隔离 v33（0.3488）下跌的原因：
  是 GAT 先验编码器本身，还是磁流掩码通道。

纪律：先验全部无样本轴（O(G+E)），无 per-sample 记忆面（v24 教训）；
先验增益由数据从零点学习。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import _slog
from model_world_h1_v25 import WorldModelH1V25


class _MaskedGraphLayer(nn.Module):
    def __init__(self, d_in: int, d_g: int = 64) -> None:
        super().__init__()
        self.q = nn.Linear(d_in, d_g, bias=False)
        self.k = nn.Linear(d_in, d_g, bias=False)
        self.v = nn.Linear(d_in, d_g, bias=False)
        self.out = nn.Linear(d_g, d_in, bias=False)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, u, src, dst, ew):
        qs = self.q(u)[src]
        ks = self.k(u)[dst]
        score = (qs * ks).sum(-1) / (qs.shape[-1] ** 0.5)
        alpha = torch.sigmoid(score) * ew
        msgs = alpha.unsqueeze(-1) * self.v(u)[dst]
        agg = torch.zeros(u.shape[0], msgs.shape[-1], device=u.device)
        agg.index_add_(0, src, msgs)
        deg = torch.zeros(u.shape[0], device=u.device)
        deg.index_add_(0, src, alpha)
        agg = agg / deg.clamp_min(1e-6).unsqueeze(-1)
        return self.norm(u + self.out(agg)), alpha


class WorldModelH1V34(WorldModelH1V25):
    def __init__(
        self,
        *args,
        n_gat_layers: int = 2,
        d_gat: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_gat_layers = int(n_gat_layers)
        self.gat = nn.ModuleList(
            [_MaskedGraphLayer(self.d_model, d_gat)
             for _ in range(self.n_gat_layers)]
        )
        self.gat_out = nn.Linear(self.d_model, self.d_z, bias=False)
        # v25 的异构聚合参数不用了，删掉
        for name in ("rel_proj", "rel_att", "rel_gate"):
            if hasattr(self, name):
                delattr(self, name)
        self._graph_ready = False

    # ------------------------------------------------------ graph build
    @torch.no_grad()
    def _build_graph(self, device) -> None:
        G = self.n_genes
        ei_list, ew_list = [], []
        for path in self.prior_graph_paths:
            g = np.load(path)
            ei = np.asarray(g["edge_index"], dtype=np.int64)
            ew = np.asarray(g["edge_w"], dtype=np.float32)
            keep = (ei[0] >= 0) & (ei[0] < G) & (ei[1] >= 0) & (ei[1] < G) \
                & (ei[0] != ei[1])
            ei_list.append(ei[:, keep])
            ew_list.append(ew[keep])
        if ei_list:
            ei = np.concatenate(ei_list, axis=1)
            ew = np.concatenate(ew_list)
            ei = np.concatenate([ei, ei[::-1]], axis=1)
            ew = np.concatenate([ew, ew])
            key = ei[0] * G + ei[1]
            uk, inv = np.unique(key, return_inverse=True)
            wmax = np.zeros(len(uk), dtype=np.float32)
            np.maximum.at(wmax, inv, ew)
            ei = np.stack([uk // G, uk % G])
            ew = wmax
        else:
            ei = np.zeros((2, 0), dtype=np.int64)
            ew = np.zeros(0, dtype=np.float32)
        src = torch.as_tensor(ei[0], dtype=torch.long, device=device)
        dst = torch.as_tensor(ei[1], dtype=torch.long, device=device)
        self.register_buffer("g_src", src)
        self.register_buffer("g_dst", dst)
        self.register_buffer(
            "g_ew", torch.sigmoid(torch.as_tensor(np.log1p(ew), device=device)))
        deg = torch.zeros(G, device=device)
        deg.index_add_(0, dst, torch.ones_like(src, dtype=torch.float32))
        self.register_buffer("g_deg", deg.clamp_min(1.0))
        self._graph_ready = True

    def _prior_encode(self, e: torch.Tensor) -> torch.Tensor:
        if not self._graph_ready:
            self._build_graph(e.device)
        h = self.prior_self(e)
        u = F.layer_norm(e, (e.shape[-1],))
        for layer in self.gat:
            u, _ = layer(u, self.g_src, self.g_dst, self.g_ew)
        return self.prior_norm(h + self.gat_out(u))

    # ------------------------------------------------------------ forward
    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        if not self._graph_ready:
            self._build_graph(device)
        B = x_ctrl.shape[0]
        e = self.gene_emb.weight
        G = e.shape[0]

        u = self._prior_encode(e)                                       # (G, d_z)
        charge_prior = self.prior_charge(u).squeeze(-1)                 # (G,)
        amp_prior = self.prior_amp(u).squeeze(-1)                       # (G,)
        src_prior = 1.0 + torch.tanh(
            self.prior_src(u[indices]).squeeze(-1))                     # (B,)

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

            dx = a * torch.tanh(r) * gate
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
        diag = super(WorldModelH1V25, self).magnetic_diagnostics(pert_idx)
        if self._graph_ready:
            diag["n_masked_edges"] = int(self.g_src.numel())
            diag["prior_inject_w"] = [
                round(float(m.weight.abs().sum()), 4)
                for m in (self.prior_charge, self.prior_amp, self.prior_src)
            ]
        return diag


WorldModel = WorldModelH1V34
