# -*- coding: utf-8 -*-
"""MagWorld v24 = v18 + 异构图先验 GNN + kq 隐空间扰动.

在 v18（v14 磁场内核 + DGIDB 手工特征双 hook）基础上新增两模块：

1. 异构图先验编码器（HeteroPriorEncoder）
   节点 = 18,533 个 panel 基因，关系类型 r ∈ {DGIDB 药物-基因互作,
   TRRUST 有向符号 TF 调控, ...}（npz: edge_index / edge_sign / edge_w）。
   每种关系有自己的投影 W_r 与 GAT 注意力 a_r，消息带符号与权重：
       m_i^r = Σ_{j∈N_r(i)} softmax_j(a_r·[W_r h_j ‖ W_r h_i]) · sign_ij · w_ij · W_r h_j
   关系级门控 β = softmax(gate) 融合各关系，加自变换残差后 LayerNorm：
       u = LayerNorm(W_self h + Σ_r β_r m^r)        (G, d_z)
   u 是"先验隐向量"：谁调控谁、谁可被药结合，全压缩进基因隐表示。

2. kq 隐空间扰动（保留磁场交互）
   扰动因子（目标基因 t）生成 query，所有基因提供 key：
       k_i = W_k u_i,   q_t = W_q u_t
       A_it = σ(k_i·q_t / √d) ∈ (0,1)      # 扰动-基因亲和门
       r_i  = w_r·k_i                        # 基因隐空间响应读出（含符号）
       resp_it = A_it ⊙ r_i                  # 隐空间扰动场
   注入点（磁场内核原样保留）：
       a) 场源调制:  phi/vec ← phi/vec · (1 + γ·A)   γ=softplus(raw_gamma)
       b) 并行残差:  h ← h + λ·resp                   λ=softplus(raw_lam)
   γ、λ 初始 softplus(-4)≈0.018 ≈ 0 ⇒ 初始函数 ≡ v18，
   init_from v18/v4 checkpoint（strict=False）行为不变，先验影响从零学起。

世界模型训练法不变（train_v14_aux.py 的 world-model loss 与 aux 混训）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import WorldModelH1V18, _slog


def _group_softmax(logit: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax within groups defined by idx (destination-node aggregation)."""
    mx = torch.full((n,), -1e30, device=logit.device, dtype=logit.dtype)
    mx = mx.index_reduce(0, idx, logit, "amax", include_self=True)
    e = torch.exp(logit - mx[idx])
    den = torch.zeros(n, device=logit.device, dtype=logit.dtype)
    den = den.index_add_(0, idx, e)
    return e / den[idx].clamp_min(1e-12)


class WorldModelH1V24(WorldModelH1V18):
    def __init__(
        self,
        *args,
        prior_graph_paths=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prior_graph_paths = [str(p) for p in (prior_graph_paths or [])]
        self.n_relations = len(self.prior_graph_paths)
        if self.n_relations == 0:
            return
        d = self.d_z  # prior latent dim = latent dim

        # ---- register heterogeneous edges as buffers ----
        for r, path in enumerate(self.prior_graph_paths):
            g = np.load(path)
            ei = torch.as_tensor(np.asarray(g["edge_index"]), dtype=torch.long)
            es = torch.as_tensor(np.asarray(g["edge_sign"]), dtype=torch.float32)
            ew = torch.as_tensor(np.asarray(g["edge_w"]), dtype=torch.float32)
            self.register_buffer("rel%d_ei" % r, ei)
            self.register_buffer("rel%d_es" % r, es)
            self.register_buffer("rel%d_ew" % r, ew)

        # ---- hetero-GNN params ----
        self.prior_self = nn.Linear(self.d_model, d, bias=False)
        self.rel_proj = nn.ModuleList(
            [nn.Linear(self.d_model, d, bias=False) for _ in range(self.n_relations)]
        )
        self.rel_att = nn.ModuleList(
            [nn.Linear(2 * d, 1, bias=False) for _ in range(self.n_relations)]
        )
        self.rel_gate = nn.Parameter(torch.zeros(self.n_relations))
        self.prior_norm = nn.LayerNorm(d)

        # ---- kq latent perturbation ----
        self.kq_key = nn.Linear(d, d, bias=False)
        self.kq_query = nn.Linear(d, d, bias=False)
        self.kq_readout = nn.Linear(d, 1, bias=False)

        # ---- fusion strengths (≈0 init: start identical to v18) ----
        self.raw_gamma = nn.Parameter(torch.tensor(-4.0))  # field-source modulation
        self.raw_lam = nn.Parameter(torch.tensor(-4.0))    # kq residual channel

    # ---------------------------------------------------------- prior encoder
    def _prior_encode(self, e: torch.Tensor) -> torch.Tensor:
        """e: frozen gene embeddings (G, d_model) -> u: prior latent (G, d_z)."""
        h = self.prior_self(e)
        msgs = []
        for r in range(self.n_relations):
            ei = getattr(self, "rel%d_ei" % r)
            es = getattr(self, "rel%d_es" % r)
            ew = getattr(self, "rel%d_ew" % r)
            src, dst = ei[0], ei[1]
            wh = self.rel_proj[r](e)                    # (G, d)
            ws, wd = wh[src], wh[dst]                   # (E, d)
            logit = self.rel_att[r](torch.cat([ws, wd], -1)).squeeze(-1)
            alpha = _group_softmax(logit, dst, e.shape[0])
            m = torch.zeros_like(h)
            m.index_add_(0, dst, (alpha * es * ew).unsqueeze(-1) * ws)
            msgs.append(m)
        beta = torch.softmax(self.rel_gate, dim=0)      # (R,)
        agg = (beta.view(-1, 1, 1) * torch.stack(msgs, 0)).sum(0)
        return self.prior_norm(h + agg)

    # ---------------------------------------------------------------- kq path
    def _kq_perturb(self, u: torch.Tensor, indices: torch.Tensor):
        """u: (G, d_z); indices: (B,) target gene idx.
        Returns affinity A (B, G) in (0,1) and latent response resp (B, G)."""
        k = self.kq_key(u)                              # (G, d) keys
        q = self.kq_query(u[indices])                   # (B, d) query
        aff = torch.sigmoid(
            (k[None, :, :] * q[:, None, :]).sum(-1) / math.sqrt(k.shape[-1])
        )                                               # (B, G)
        readout = self.kq_readout(k).squeeze(-1)        # (G,)
        return aff, aff * readout[None, :]

    # ------------------------------------------------------------- main path
    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        if self.n_relations == 0:
            return super().predict_delta(x_ctrl, pert_idx)
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        B = x_ctrl.shape[0]
        e = self.gene_emb.weight
        G = e.shape[0]

        # ---- hetero-prior latent + kq perturbation field ----
        u = self._prior_encode(e)                       # (G, d_z)
        aff, resp = self._kq_perturb(u, indices)        # (B, G) each
        gamma = F.softplus(self.raw_gamma)
        lam = F.softplus(self.raw_lam)

        pos = F.normalize(self.pole(e), dim=-1)
        charge = F.softplus(self.charge(e).squeeze(-1) + self.charge_bias)

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

        # ---- v18 DGIDB hooks ----
        f = getattr(self, "dgidb_feat", None)                           # (G, F) or None
        if f is not None:
            f_src = f[indices]                                          # (B, F)
            src_gain = 1.0 + torch.tanh(self.src_proj(f_src)).squeeze(-1)   # (B,)
            amp_bias = self.amp_dg(f).squeeze(-1)                       # (G,)
        else:
            src_gain = torch.ones(B, device=device)
            amp_bias = torch.zeros(G, device=device)

        # ---- source injection with kq field modulation (magnetic core kept) ----
        boost = 1.0 + gamma * aff                                       # (B, G)
        phi = kappa * charge[indices][:, None] * k_src * src_gain[:, None] * boost
        vec = (phi)[:, :, None] * u_src

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

        # ---- kq latent residual channel (parallel to magnetic pathway) ----
        h = h + lam * resp

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
        if self.n_relations > 0:
            diag["n_relations"] = int(self.n_relations)
            diag["relation_names"] = [
                str(p).split("/")[-1] for p in self.prior_graph_paths
            ]
            diag["relation_beta"] = [
                round(float(v), 4) for v in torch.softmax(self.rel_gate, dim=0)
            ]
            diag["kq_gamma"] = float(F.softplus(self.raw_gamma))
            diag["kq_lambda"] = float(F.softplus(self.raw_lam))
        return diag


WorldModel = WorldModelH1V24
