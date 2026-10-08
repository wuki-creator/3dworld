
# -*- coding: utf-8 -*-
"""
MagWorld v17 "MagFluid" — v13 magnetic field + porous-medium fluid transport.

v13 (unchanged, ablation-proven): scf_encoder basal context, magnetic force
rollout, interaction bilinear, z_delta modulation, self effect, tanh bound.

NEW fluid layer: after the magnetic pair_field is computed, it is redistributed
by ONE porous-medium (m=2) transport step on the gene graph:

    porous medium equation:   d c / d tau = div( K(c) grad c )
    implemented as upwind flux F_ij = K_ij (v_j - v_i),  v = c * |c|

    K_ij = (1 + a_i . b_j / d_z)^2      (permeability from gene embeddings,
                                         nonnegative by construction,
                                         factorized O(G d_z) application)
    c' = (1 - dt) c + dt * (K v) / (K 1)    (random-walk normalization:
                                             convex average, unconditionally
                                             stable for dt in (0,1))

Why porous medium and not plain diffusion / quantum waves:
  * quadratic flux (flux ~ grad c^2) => finite propagation speed and
    self-sharpening fronts (Barenblatt solutions) — response stays LOCAL
    while being transported along embedding-similarity pathways;
  * mass conservation: transport only relocates the magnetic response,
    it never invents magnitude (nmae-friendly);
  * sign carried by donor (v = c*|c|): up/down regulation pattern is
    preserved through transport.

The transported field is blended with the untransported one via a learned
sigmoid (fl_mix), so training can shut the fluid off if it does not help.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v13 import WorldModelH1V13


class WorldModelH1V17(WorldModelH1V13):
    """v13 magnetic backbone + porous-medium fluid redistribution."""

    FL_STEPS: int = 2  # transport iterations per forward

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = self.d_model
        if hasattr(self, "context_encoder"):      # dead Chebyshev params
            del self.context_encoder
        # permeability field from the shared gene table
        self.fl_c = nn.Linear(d_model, self.d_z, bias=False)   # receiver side
        self.fl_o = nn.Linear(d_model, self.d_z, bias=False)   # donor side
        self.fl_dt_raw = nn.Parameter(torch.tensor(0.0))       # dt in (0,1)
        self.fl_mix = nn.Parameter(torch.tensor(0.0))          # blend weight

    def _fluid(self, embeddings: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Porous-medium transport of the signed magnetic impulse m. (B, G)"""
        A = self.fl_c(embeddings) / math.sqrt(self.d_z)        # (G, d_z)
        Bm = self.fl_o(embeddings) / math.sqrt(self.d_z)       # (G, d_z)
        A2 = A * A
        B2 = Bm * Bm
        G = m.shape[1]
        # row sums of K = (1 + a.b)^2 :  K1 = G + 2 A (sum B) + A2 (sum B2)
        ones = torch.ones(1, G, dtype=m.dtype, device=m.device)
        sb = ones @ Bm                                           # (1, d_z)
        sb2 = ones @ B2                                          # (1, d_z)
        k_row = (G + 2.0 * (sb @ A.T) + (sb2 @ A2.T))[0]         # (G,)
        k_row = k_row.clamp(min=1e-3)

        dt = torch.sigmoid(self.fl_dt_raw)
        c = m
        for _ in range(self.FL_STEPS):
            v = c * c.abs()                                      # signed c*|c|
            # w_i = sum_j K_ij v_j  with K=(1+a.b)^2 = 1 + 2ab + (ab)^2
            w = (
                v.sum(1, keepdim=True)
                + 2.0 * ((v @ Bm) @ A.T)
                + ((v @ B2) @ A2.T)
            )
            c = (1.0 - dt) * c + dt * w / k_row[None, :]
        return c

    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        indices = self._indices(pert_idx).to(x_ctrl.device)
        embeddings = self.gene_emb.weight

        # (3) v6 magnetic field (UNCHANGED)
        magnetic_force, _, _ = self.magnet(embeddings, indices)
        source = self.source_code(embeddings[indices])
        receiver = self.receiver_code(embeddings) + self.receiver_residual
        interaction = source @ receiver.T / math.sqrt(self.d_z)

        # (1)+(2) decoupled latents (v13, unchanged)
        context_input = (
            F.layer_norm(x_ctrl, (self.n_genes,))
            if self.normalize_context else x_ctrl
        )
        z_basal = self.scf_encoder(context_input)                # (B, D)
        pert_2d = indices.unsqueeze(1) if indices.dim() == 1 else indices
        z_delta = self.pert_encoder(pert_2d)                     # (B, D)

        # (4) basal state -> context gate (unchanged)
        context = self.zbasal_proj(z_basal)                      # (B, d_z)
        context_receiver = F.normalize(self.context_receiver(embeddings), dim=-1)
        context_gate = self.context_strength * torch.tanh(
            context @ context_receiver.T / math.sqrt(self.d_z))

        magnetic_weight = torch.sigmoid(self.magnetic_mix)
        pair_field = magnetic_weight * magnetic_force + (1.0 - magnetic_weight) * interaction

        # (3.5) NEW: porous-medium fluid redistribution of the magnetic field
        transported = self._fluid(embeddings, pair_field)
        fluid_weight = torch.sigmoid(self.fl_mix)
        pair_field = fluid_weight * transported + (1.0 - fluid_weight) * pair_field

        shared_bias_scale = self.effective_shared_bias_scale()
        raw_delta = (
            F.softplus(self.gene_scale)[None, :] * pair_field
            + shared_bias_scale * self.gene_bias[None, :]
        ) * (1.0 + context_gate)

        # (5) z_delta per-gene modulation (unchanged)
        zd = self.zdelta_proj(torch.sigmoid(self.latent_gate) * z_delta)
        mod = zd @ receiver.T / math.sqrt(self.d_z)
        raw_delta = raw_delta + F.softplus(self.zdelta_scale) * torch.tanh(mod)

        # (6) self effect + bounded output (unchanged)
        self_effect = -F.softplus(self.raw_self_effect)
        raw_delta = raw_delta.scatter_add(
            1, indices[:, None], self_effect.expand(len(indices), 1))
        return self.max_delta * torch.tanh(raw_delta / self.max_delta)


WorldModel = WorldModelH1V17
