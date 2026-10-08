# -*- coding: utf-8 -*-
"""MagWorld v20 = v14 + DGIDB prior with GAT structural extraction.

Prior path:
  1. Count features f_i (8-dim, from DGIDB interactions, standardized).
  2. Gene-gene co-drug graph (genes sharing drugs, signed by interaction
     class, promiscuous drugs down-weighted 1/sqrt(k)). A 2-layer multi-head
     graph attention network (GAT) extracts structural embedding p_i (16-dim)
     from f and the graph topology; attention logits are biased by edge sign
     (learnable per-class scalar) and log edge weight.
  3. Prior vector u_i = [f_i ; p_i] (24-dim) drives the two v18 hooks:
       src_gain = 1 + tanh(w_src . u_target)      (perturbation source strength)
       amp_bias = w_amp . u_i                      (per-gene susceptibility)
     Output projections are zero-initialized -> the prior starts neutral and
     can only help; base v14 behavior is unchanged at init.

Gene-gene graph edges are static buffers saved in the checkpoint, so a v20
checkpoint is self-contained for prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v14 import WorldModelH1V14, _slog

_GAT_HEADS_L1 = 4
_GAT_DIM_L1 = 8
_GAT_HEADS_L2 = 4
_GAT_DIM_L2 = 4
_GAT_OUT = _GAT_HEADS_L2 * _GAT_DIM_L2   # 16


class _GATLayer(nn.Module):
    """Single GAT layer with additive edge-sign bias and log-weight bias."""

    def __init__(self, d_in: int, heads: int, d_head: int) -> None:
        super().__init__()
        self.heads = heads
        self.d_head = d_head
        self.q = nn.Linear(d_in, heads * d_head, bias=False)
        self.k = nn.Linear(d_in, heads * d_head, bias=False)
        self.v = nn.Linear(d_in, heads * d_head, bias=False)
        self.sign_bias = nn.Parameter(torch.zeros(2))  # [neg, pos]
        self.res = nn.Linear(d_in, heads * d_head, bias=False)

    def forward(self, h, src, dst, sign_idx, log_w):
        G = h.shape[0]
        q = self.q(h).view(G, self.heads, self.d_head)
        k = self.k(h).view(G, self.heads, self.d_head)
        v = self.v(h).view(G, self.heads, self.d_head)
        logits = (q[dst] * k[src]).sum(-1) / (self.d_head ** 0.5)
        logits = logits + (self.sign_bias[sign_idx] + log_w)[:, None]
        # masked softmax over incoming edges per (dst, head)
        m = torch.full((G, self.heads), -1e9, device=h.device)
        m = m.index_reduce(0, dst, logits, "amax", include_self=True)
        alpha = torch.exp(logits - m[dst])
        denom = torch.zeros(G, self.heads, device=h.device)
        denom = denom.index_add_(0, dst, alpha)
        alpha = alpha / (denom[dst] + 1e-8)
        msg = alpha.unsqueeze(-1) * v[src]                    # (E, heads, d)
        out = torch.zeros(G, self.heads, self.d_head, device=h.device)
        out = out.index_add_(0, dst, msg)
        return out.reshape(G, self.heads * self.d_head) + self.res(h)


class WorldModelH1V20(WorldModelH1V14):
    def __init__(self, *args, n_dgidb_feats: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_dgidb_feats = int(n_dgidb_feats)
        if self.n_dgidb_feats > 0:
            prior_dim = self.n_dgidb_feats + _GAT_OUT
            self.src_proj = nn.Linear(prior_dim, 1, bias=False)
            self.amp_dg = nn.Linear(prior_dim, 1, bias=False)
            nn.init.zeros_(self.src_proj.weight)
            nn.init.zeros_(self.amp_dg.weight)
            self.register_buffer(
                "dgidb_feat", torch.zeros(self.n_genes, self.n_dgidb_feats)
            )
            self.register_buffer("dgidb_std", torch.ones(self.n_dgidb_feats))
            self.gat_in = nn.Sequential(
                nn.Linear(self.n_dgidb_feats, _GAT_HEADS_L1 * _GAT_DIM_L1),
                nn.GELU(),
            )
            self.gat1 = _GATLayer(_GAT_HEADS_L1 * _GAT_DIM_L1, _GAT_HEADS_L1, _GAT_DIM_L1)
            self.gat_norm1 = nn.LayerNorm(_GAT_HEADS_L1 * _GAT_DIM_L1)
            self.gat2 = _GATLayer(_GAT_HEADS_L1 * _GAT_DIM_L1, _GAT_HEADS_L2, _GAT_DIM_L2)
            # static graph (filled by load_dgidb_graph); empty = neutral
            self.register_buffer(
                "dgidb_edge_index", torch.zeros(2, 0, dtype=torch.long)
            )
            self.register_buffer("dgidb_edge_sign", torch.zeros(0))
            self.register_buffer("dgidb_edge_w", torch.zeros(0))

    @torch.no_grad()
    def load_dgidb_prior(self, feats: np.ndarray) -> None:
        f = torch.as_tensor(feats, dtype=torch.float32)
        if f.shape != self.dgidb_feat.shape:
            raise ValueError("dgidb features %s != %s"
                             % (tuple(f.shape), tuple(self.dgidb_feat.shape)))
        std = f.std(0).clamp_min(1e-3)
        self.dgidb_std.copy_(std)
        self.dgidb_feat.copy_((f - f.mean(0)) / std)

    @torch.no_grad()
    def load_dgidb_graph(self, edge_index: np.ndarray, edge_sign: np.ndarray,
                         edge_w: np.ndarray) -> None:
        ei = torch.as_tensor(edge_index, dtype=torch.long)
        if ei.max() >= self.n_genes:
            raise ValueError("graph index out of panel range")
        self.dgidb_edge_index = ei.to(self.dgidb_feat.device)
        self.dgidb_edge_sign = torch.as_tensor(edge_sign, dtype=torch.float32).to(self.dgidb_feat.device)
        self.dgidb_edge_w = torch.as_tensor(edge_w, dtype=torch.float32).to(self.dgidb_feat.device)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # graph buffers are resized at load time (default is empty), so adopt
        # their shapes from the checkpoint instead of mismatching.
        for name in ("dgidb_edge_index", "dgidb_edge_sign", "dgidb_edge_w"):
            if name in state_dict:
                setattr(self, name, state_dict[name])
        return super().load_state_dict(state_dict, *args, **kwargs)

    def prior_vectors(self):
        """Return (u_all (G, F+P), src/amp driven later)."""
        f = self.dgidb_feat
        if self.dgidb_edge_index.shape[1] == 0:
            p = torch.zeros(self.n_genes, _GAT_OUT, device=f.device)
        else:
            h = self.gat_in(f)
            src = self.dgidb_edge_index[0]
            dst = self.dgidb_edge_index[1]
            sign_idx = (self.dgidb_edge_sign > 0).long()
            log_w = torch.log1p(self.dgidb_edge_w)
            h = self.gat_norm1(F.gelu(self.gat1(h, src, dst, sign_idx, log_w)))
            p = self.gat2(h, src, dst, sign_idx, log_w)
        return torch.cat([f, p], dim=-1)

    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        if self.n_dgidb_feats == 0:
            return super().predict_delta(x_ctrl, pert_idx)
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        u = self.prior_vectors()                                        # (G, F+P)
        src_gain = 1.0 + torch.tanh(self.src_proj(u[indices])).squeeze(-1)  # (B,)
        amp_bias = self.amp_dg(u).squeeze(-1)                           # (G,)

        B = x_ctrl.shape[0]
        e = self.gene_emb.weight
        G = e.shape[0]

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
        phi = kappa * charge[indices][:, None] * k_src * src_gain[:, None]
        vec = (kappa * charge[indices][:, None] * k_src * src_gain[:, None])[:, :, None] * u_src

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


WorldModel = WorldModelH1V20
