# -*- coding: utf-8 -*-
"""MagWorld v32 = 无磁流框架 + DGIDB 掩码图注意力先验.

与 v18/v25 的本质区别：没有任何磁流件（磁极/磁荷/1/r^α 核/磁矩/
朗之万/弛豫 rollout 全部不用）。扰动传播的载体就是 DGIDB 图结构本身：

  1. 掩码图注意力（先验编码器，per-gene 静态，无样本轴）:
       基因节点 = scFoundation/hybrid 嵌入 e
       掩码 M = DGIDB 边（双向化 + 自环）
       只允许沿 M 的边算注意力：score_ij = (Wq u_i)·(Wk u_j)/√d
       α_ij = sigmoid(score_ij)·σ(w_ij)      # w_ij = 先验边权
       u_i ← LayerNorm(u_i + Σ α_ij Wv u_j / (Σ α_ij + ε))，2 层

  2. p(target gene | source gene):
       源节点 s 的归一化注意力行 α_s·/Σα_s —— 即「扰动 s 影响各靶点」
       的条件分布，直接作为响应场的强度输入 φ（reach）。

  3. 样本响应（保留 v18 幅度/取向解耦头与输出端）:
       z = context_encoder(x_ctrl)（沿用 v4 底座）
       s = state_gene(e) + state_ctx(z)
       c_src = u[indices]（源的先验语境嵌入，broadcast）
       pre_a = amp_gene(s) + amp_ctx(c_src) + reach_gate·slog(φ) + amp_bias
       a = σ(pre_a)；d = normalize(dir_proj(s))；g = σ(W d + b)
       Δx = a ⊙ tanh(R(φ)) ⊙ g
       + 共享偏置 ⊙ (1+上下文门控) + 自效应，max_delta tanh 限幅

纪律（v24 教训）：图编码器无样本轴，先验只能调制基因静态属性；
reach_gate 零初始化 → 初始 ≈ 无先验基线，先验强度由数据决定。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import WorldModelH1V18, _slog


class _MaskedGraphLayer(nn.Module):
    def __init__(self, d_in: int, d_g: int = 64) -> None:
        super().__init__()
        self.q = nn.Linear(d_in, d_g, bias=False)
        self.k = nn.Linear(d_in, d_g, bias=False)
        self.v = nn.Linear(d_in, d_g, bias=False)
        self.out = nn.Linear(d_g, d_in, bias=False)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, u: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                ew: torch.Tensor) -> torch.Tensor:
        qs = self.q(u)[src]                       # (E, d_g)
        ks = self.k(u)[dst]                       # (E, d_g)
        score = (qs * ks).sum(-1) / (qs.shape[-1] ** 0.5)
        alpha = torch.sigmoid(score) * ew         # (E,) 先验边权调制
        msgs = alpha.unsqueeze(-1) * self.v(u)[dst]   # (E, d_g)
        agg = torch.zeros(u.shape[0], msgs.shape[-1], device=u.device)
        agg.index_add_(0, src, msgs)
        deg = torch.zeros(u.shape[0], device=u.device)
        deg.index_add_(0, src, alpha)
        agg = agg / deg.clamp_min(1e-6).unsqueeze(-1)
        return self.norm(u + self.out(agg))


class WorldModelH1V32(WorldModelH1V18):
    def __init__(
        self,
        *args,
        prior_graph_paths=None,
        n_gat_layers: int = 2,
        d_gat: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prior_graph_paths = [str(p) for p in (prior_graph_paths or [])]
        self.n_gat_layers = int(n_gat_layers)
        d = self.d_model
        self.gat = nn.ModuleList(
            [_MaskedGraphLayer(d, d_gat) for _ in range(self.n_gat_layers)]
        )
        self.src_ctx_proj = nn.Linear(d, self.d_dir, bias=False)
        # 先验场进入幅度的门（零初始化 → 初始 ≡ 无先验）
        self.raw_reach_gate = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.src_ctx_proj.weight)
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
            # 双向化
            ei = np.concatenate([ei, ei[::-1]], axis=1)
            ew = np.concatenate([ew, ew])
            # 去重取最大权
            key = ei[0] * G + ei[1]
            uk, inv = np.unique(key, return_inverse=True)
            wmax = np.zeros(len(uk), dtype=np.float32)
            np.maximum.at(wmax, inv, ew)
            ei = np.stack([uk // G, uk % G])
            ew = wmax
        else:
            ei = np.zeros((2, 0), dtype=np.int64)
            ew = np.zeros(0, dtype=np.float32)
        self.register_buffer("g_src", torch.as_tensor(ei[0], device=device))
        self.register_buffer("g_dst", torch.as_tensor(ei[1], device=device))
        # 先验边权压到 (0,1) 作 sigmoid 前的乘性偏置
        self.register_buffer(
            "g_ew", torch.sigmoid(torch.as_tensor(np.log1p(ew), device=device)))
        self._graph_ready = True

    def _prior_encode(self, e: torch.Tensor) -> torch.Tensor:
        u = F.layer_norm(e, (e.shape[-1],))
        for layer in self.gat:
            u = layer(u, self.g_src, self.g_dst, self.g_ew)
        return u

    # ------------------------------------------------------------ forward
    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        if not self._graph_ready:
            self._build_graph(device)
        B = x_ctrl.shape[0]
        G = self.n_genes
        e = self.gene_emb.weight

        u = self._prior_encode(e)                                     # (G, d)
        ctx_in = F.layer_norm(x_ctrl, (G,)) if self.normalize_context else x_ctrl
        z = self.context_encoder(ctx_in)
        s = self.state_gene(e)[None, :, :] + self.state_ctx(z)[:, None, :]
        c_src = self.src_ctx_proj(u[indices])                         # (B, d_dir)
        s = s + c_src[:, None, :]

        # p(target|source)：源节点注意力行的归一化（掩码图上的可达分布）
        with torch.no_grad():
            alpha_rows = []
            ul = F.layer_norm(e, (e.shape[-1],))
            for layer in self.gat:
                qs = layer.q(ul)[self.g_src]
                ks = layer.k(ul)[self.g_dst]
                score = (qs * ks).sum(-1) / (qs.shape[-1] ** 0.5)
                alpha_rows.append(torch.sigmoid(score) * self.g_ew)
            alpha = alpha_rows[-1]
        reach = torch.zeros(B, G, device=device)
        for b in range(B):
            m = self.g_src == int(indices[b])
            if bool(m.any()):
                reach[b].index_add_(0, self.g_dst[m], alpha[m])
        reach = reach / reach.sum(1, keepdim=True).clamp_min(1e-8)

        f = getattr(self, "dgidb_feat", None)
        amp_bias = self.amp_dg(f).squeeze(-1) if f is not None \
            else torch.zeros(G, device=device)

        reach_gate = self.raw_reach_gate
        phi = reach + 1e-6
        h_ext = phi.mean(1, keepdim=True)
        pre_a = (
            self.amp_gene(s).squeeze(-1)
            + reach_gate * _slog(phi)
            + self.amp_phi(_slog(phi).unsqueeze(-1)).squeeze(-1)
            + self.amp_v(_slog(phi).unsqueeze(-1)).squeeze(-1)
            + self.amp_ext(h_ext)
            + self.amp_bias
            + amp_bias[None, :]
        )
        a = torch.sigmoid(pre_a)

        d = F.normalize(self.dir_proj(s), dim=-1)
        gate = torch.sigmoid(self.gate_proj(d).squeeze(-1) + self.gate_bias)

        phi_n = _slog(phi)
        phi_n = phi_n / phi_n.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        r = self.response_right(phi_n) @ self.response_left.weight

        h = a * torch.tanh(r) * gate

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
        diag = {
            "n_prior_edges": int(self.g_src.numel()) if self._graph_ready else 0,
            "reach_gate": round(float(self.raw_reach_gate), 4),
            "n_gat_layers": self.n_gat_layers,
        }
        return diag


WorldModel = WorldModelH1V32
