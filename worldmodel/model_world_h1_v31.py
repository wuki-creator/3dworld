# -*- coding: utf-8 -*-
"""MagWorld v31 = 张力模型 (TensionWorld)：scFoundation 底座 + 弹性网络张力传播.

对应用户定义的三要素：
  p(target gene | source gene)：归一化张力可达矩阵 P = T / row_sum(T)，
      源基因 s 的一行 P[s] 即「扰动 s 后影响各靶基因」的条件分布。
  张力约束公式：弹性网络能量 E(Δx) = ½ Σ_ij T_ij (Δx_i − Δx_j)² = Δxᵀ L Δx，
      扰动响应 = 锚定源位移 δ 后的调和（谐波）平衡解：
      Δx* = argmin E(Δx)  s.t. Δx_s = δ   →  Jacobi 松弛迭代可微近似。
  基因互作先验网络：DGIDB / TRRUST / STRING（--prior-graphs 传入，
      与 v25 同格式 npz：edge_index/edge_w），每条边叠乘 (1+Σ_r β_r A_r)。
  基因结构先验：STRING 通道（含 fusion/co-occurrence 结构证据），
      边权几何由 scFoundation 基因嵌入（--gene-embeddings scfound_pca256.npy）决定。

与 v25 相同的纪律：几何固定（首 forward 惰性构图，detach），
可学习的只有张力标量（长度尺度 γ、每关系 β_r）+ v18 全套幅度/取向头。

前向（替换 v18 的磁核几何，响应头/共享偏置/自效应/限幅全部沿用）：
  T = exp(-γ·d²) ⊙ (1 + Σ_r softplus(β_r)·A_r)     # 边张力
  u ← 上下文力场 m = charge⊙(1+tanh(w·expr+b))      # 背景张力位移
  n_layers 轮锚定松弛: u_i ← Σ_j T_ij u_j / Σ_j T_ij,  u_s ≡ δ
  响应 = a(|u|, s) ⊙ tanh(R|u|) ⊙ gate(取向头 + η·tanh(u))
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import WorldModelH1V18, _slog


class WorldModelH1V31(WorldModelH1V18):
    def __init__(
        self,
        *args,
        prior_graph_paths=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prior_graph_paths = [str(p) for p in (prior_graph_paths or [])]
        self.n_relations = len(self.prior_graph_paths)
        # 张力标量（低自由度）
        self.raw_inv_ell = nn.Parameter(torch.tensor(0.5413))          # softplus≈1.0
        if self.n_relations:
            self.raw_rel_beta = nn.Parameter(
                torch.full((self.n_relations,), -0.4328))              # softplus≈0.5
        self._edges_ready = False

    # ------------------------------------------------------ graph build
    @torch.no_grad()
    def _build_edges(self, device) -> None:
        G = self.n_genes
        e = F.normalize(self.gene_emb.weight.detach(), dim=-1)         # (G, d)
        # kNN on scFoundation geometry
        sim = e @ e.T                                                   # (G, G)
        k = int(getattr(self, "graph_k", 50))
        nb = sim.topk(k + 1, dim=1).indices[:, 1:]                      # 去掉自环
        src = torch.arange(G, device=device)[:, None].expand(-1, k).reshape(-1)
        dst = nb.reshape(-1)
        ei = torch.stack([src, dst]).cpu().numpy()
        d_np = (2.0 - 2.0 * sim.cpu().numpy()[src.cpu().numpy(),
                                             dst.cpu().numpy()]).clip(min=0.0)
        # prior edges（与 kNN 取并集）——全部向量化，不用 Python 行循环
        a_list = []
        if self.n_relations:
            key_ei = ei[0].astype(np.int64) * G + ei[1].astype(np.int64)
            order = np.argsort(key_ei)
            sorted_keys = key_ei[order]
            a_knn = np.zeros((self.n_relations, ei.shape[1]), dtype=np.float32)
            CAP = 1_000_000  # 每关系边数上限（按权截断）
            for r, path in enumerate(self.prior_graph_paths):
                g = np.load(path)
                pei = np.asarray(g["edge_index"], dtype=np.int64)
                pw = np.abs(np.asarray(g["edge_w"], dtype=np.float32))
                keep = (pei[0] >= 0) & (pei[0] < G) & (pei[1] >= 0) & (pei[1] < G)
                pei, pw = pei[:, keep], pw[keep]
                if pei.shape[1] > CAP:
                    top = np.argpartition(pw, -CAP)[-CAP:]
                    pei, pw = pei[:, top], pw[top]
                key_pei = pei[0] * G + pei[1]
                pos_in_sorted = np.clip(
                    np.searchsorted(sorted_keys, key_pei), 0, len(sorted_keys) - 1)
                found = sorted_keys[pos_in_sorted] == key_pei
                np.maximum.at(a_knn[r], order[pos_in_sorted[found]], pw[found])
                extra, extra_w = pei[:, ~found], pw[~found]
                if extra.shape[1]:
                    ea = e.cpu().numpy()
                    diff = (ea[extra[0]] - ea[extra[1]]).astype(np.float32)
                    ed = np.einsum("ij,ij->i", diff, diff).clip(min=0.0)
                    ei = np.concatenate([ei, extra], axis=1)
                    d_np = np.concatenate([d_np, ed])
                    grow = np.zeros((self.n_relations, extra.shape[1]),
                                    dtype=np.float32)
                    grow[r] = extra_w
                    a_knn = np.concatenate([a_knn, grow], axis=1)
            a_list = [a_knn]
        # 去重
        key = ei[0].astype(np.int64) * G + ei[1].astype(np.int64)
        _, uniq = np.unique(key, return_index=True)
        self.register_buffer("t_src", torch.as_tensor(
            ei[0][uniq], dtype=torch.long, device=device))
        self.register_buffer("t_dst", torch.as_tensor(
            ei[1][uniq], dtype=torch.long, device=device))
        self.register_buffer("t_d2", torch.as_tensor(
            d_np[uniq], dtype=torch.float32, device=device))
        if a_list:
            self.register_buffer("t_A", torch.as_tensor(
                a_list[0][:, uniq], dtype=torch.float32, device=device))
        self._edges_ready = True

    # ------------------------------------------------------------- kernel
    def _tension(self) -> torch.Tensor:
        """T = exp(-γ·d²) ⊙ (1 + Σ_r softplus(β_r)·A_r)，(E,)"""
        gamma = F.softplus(self.raw_inv_ell)
        t = torch.exp(-gamma * self.t_d2)
        if self.n_relations:
            beta = F.softplus(self.raw_rel_beta)
            t = t * (1.0 + (beta[:, None] * self.t_A).sum(0))
        return t

    @torch.no_grad()
    def tension_reach(self, src_idx) -> torch.Tensor:
        """p(target|source)：行归一化张力可达矩阵的源行。(B, G)"""
        if not self._edges_ready:
            self._build_edges(self.gene_emb.weight.device)
        t = self._tension()
        row = torch.zeros(self.n_genes, device=t.device)
        row.index_add_(0, self.t_dst, t)
        p = torch.zeros(len(src_idx), self.n_genes, device=t.device)
        for b, s in enumerate(src_idx.tolist()):
            m = self.t_src == s
            p[b].index_add_(0, self.t_dst[m], t[m])
            p[b] /= row.clamp_min(1e-8)
        return p

    # ------------------------------------------------------------ forward
    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        if not self._edges_ready:
            self._build_edges(device)
        B = x_ctrl.shape[0]
        G = self.n_genes
        e = self.gene_emb.weight

        charge = F.softplus(self.charge(e).squeeze(-1) + self.charge_bias)
        ctx_in = F.layer_norm(x_ctrl, (G,)) if self.normalize_context else x_ctrl
        z = self.context_encoder(ctx_in)
        s = self.state_gene(e)[None, :, :] + self.state_ctx(z)[:, None, :]

        expr = torch.log1p(x_ctrl.clamp_min(0.0))
        moment = charge[None, :] * (
            1.0 + torch.tanh(self.moment_expr_w * expr + self.moment_expr_b)
        )

        # ---- 张力场 ----
        t = self._tension()                                             # (E,)
        row = torch.zeros(G, device=device)
        row.index_add_(0, self.t_dst, t)
        row = row.clamp_min(1e-8)

        # 源锚定力位移 δ = κ·charge_s·src_gain
        kappa = F.softplus(self.raw_source_kappa)
        f = getattr(self, "dgidb_feat", None)
        if f is not None:
            f_src = f[indices]
            src_gain = 1.0 + torch.tanh(self.src_proj(f_src)).squeeze(-1)
            amp_bias = self.amp_dg(f).squeeze(-1)
        else:
            src_gain = torch.ones(B, device=device)
            amp_bias = torch.zeros(G, device=device)
        delta = kappa * charge[indices] * src_gain                      # (B,)

        # 显式 p(target|source) 直达注入：归一化张力可达行
        matches = self.t_src[None, :] == indices[:, None]               # (B, E)
        p_reach = torch.zeros(B, G, device=device).index_add_(
            1, self.t_dst, matches.float() * t[None, :])
        p_reach = p_reach / row[None, :]

        # 锚定调和松弛：u ← 上下文力场 + δ·p_reach，再松弛 u_i ← Σ_j T_ij u_j / Σ_j T_ij
        u = moment + delta[:, None] * p_reach
        is_src = torch.zeros(B, G, dtype=torch.bool, device=device)
        is_src.scatter_(1, indices[:, None], True)
        for _ in range(self.n_layers):
            contrib = t[None, :] * u[:, self.t_src]
            u_new = torch.zeros_like(u).index_add_(1, self.t_dst, contrib)
            u_new = u_new / row[None, :]
            u = torch.where(is_src, delta[:, None].expand(-1, G), u_new)

        # ---- 响应（幅度/取向解耦，沿用 v18 头）----
        phi = u.abs() + 1e-6
        v_norm = phi
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
        eta = torch.sigmoid(self.raw_eta) * 2.0
        d = F.normalize(d0 + eta * torch.tanh(u).unsqueeze(-1), dim=-1)
        gate = torch.sigmoid(self.gate_proj(d).squeeze(-1) + self.gate_bias)

        phi_n = _slog(phi)
        phi_n = phi_n / phi_n.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        r = self.response_right(phi_n) @ self.response_left.weight

        dx = a * torch.tanh(r) * gate
        h = dx

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
        if self._edges_ready:
            diag["n_tension_edges"] = int(self.t_src.numel())
            diag["inv_ell"] = round(float(F.softplus(self.raw_inv_ell)), 4)
            if self.n_relations:
                diag["rel_beta"] = [
                    round(float(v), 4)
                    for v in F.softplus(self.raw_rel_beta)
                ]
                diag["relation_names"] = [
                    p.split("/")[-1] for p in self.prior_graph_paths
                ]
            try:
                p_reach = self.tension_reach(
                    self._indices(pert_idx)[:8].to(self.t_src.device))
                ent = -(p_reach.clamp_min(1e-12)
                        * p_reach.clamp_min(1e-12).log()).sum(1)
                diag["reach_entropy_mean"] = round(float(ent.mean()), 3)
            except Exception:
                pass
        return diag


WorldModel = WorldModelH1V31
