# -*- coding: utf-8 -*-
"""MagWorld v18 = v14 + DGIDB drug-gene interaction prior.

Prior injection points (features: per-gene log1p weighted counts over
inhibitor/blocker/antibody/negative-modulator vs agonist/activator/positive-
modulator/potentiator classes, approved drugs weighted 2x):

  1. Source-side (target gene): the perturbation source strength is scaled by
         src_gain = 1 + tanh(w_src . f_target)   in (0, 2)
     A gene with a rich drug-inhibition history (druggable, tightly regulated)
     gets a stronger/sharper injected perturbation field.
  2. Gene-side (response amplitude): per-gene susceptibility bias
         pre_a_i += w_amp . f_i
     Genes that drugs bind tend to be downstream-regulated genes; this biases
     their response amplitude. Genes without DGIDB records (f_i = 0) are
     unaffected.

Everything else identical to WorldModelH1V14 (same state dict prefix, so
v4_seed337 / v14 checkpoints load with strict=False).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v14 import WorldModelH1V14, _slog


class WorldModelH1V18(WorldModelH1V14):
    def __init__(self, *args, n_dgidb_feats: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_dgidb_feats = int(n_dgidb_feats)
        if self.n_dgidb_feats > 0:
            self.src_proj = nn.Linear(self.n_dgidb_feats, 1, bias=False)
            self.amp_dg = nn.Linear(self.n_dgidb_feats, 1, bias=False)
            nn.init.zeros_(self.src_proj.weight)
            nn.init.zeros_(self.amp_dg.weight)
            self.register_buffer(
                "dgidb_feat", torch.zeros(self.n_genes, self.n_dgidb_feats)
            )
            self.register_buffer("dgidb_std", torch.ones(self.n_dgidb_feats))

    @torch.no_grad()
    def load_dgidb_prior(self, feats: np.ndarray) -> None:
        f = torch.as_tensor(feats, dtype=torch.float32)
        if f.shape != self.dgidb_feat.shape:
            raise ValueError("dgidb features %s != %s"
                             % (tuple(f.shape), tuple(self.dgidb_feat.shape)))
        std = f.std(0).clamp_min(1e-3)
        self.dgidb_std.copy_(std)
        self.dgidb_feat.copy_((f - f.mean(0)) / std)

    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        if self.n_dgidb_feats == 0:
            return super().predict_delta(x_ctrl, pert_idx)
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        f = self.dgidb_feat                                             # (G, F)
        f_src = f[indices]                                              # (B, F)
        src_gain = 1.0 + torch.tanh(self.src_proj(f_src)).squeeze(-1)   # (B,)
        amp_bias = self.amp_dg(f).squeeze(-1)                           # (G,)

        # ---- replicate the v14 forward with the two prior hooks ----
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
        # ---- prior hook 1: target-side source gain ----
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
                + amp_bias[None, :]          # ---- prior hook 2: gene-side amplitude
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


WorldModel = WorldModelH1V18
