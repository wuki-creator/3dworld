
# -*- coding: utf-8 -*-
"""
MagWorld v13 — hybrid: v12's decoupled dual-latent grafted onto the v6/v7
magnetic-field backbone.

Algorithm (predict_delta):
  1. z_basal = ScFoundationEncoder(LN(x_ctrl))         # basal state (v12)
  2. z_delta = MagneticPerturbEncoder(pert)            # perturbation latent (v12:
                                                       # reciprocal field rollout
                                                       # + identity pathway)
  3. per-gene magnetic field (v6 backbone, UNCHANGED):
       magnetic_force from K-step directed reciprocal rollout;
       interaction = source_code(pert) @ receiver_code(all genes)';
       pair_field = sigma(mix) * force + (1-sigma(mix)) * interaction
  4. basal state enters as the per-gene CONTEXT GATE (replacing v6's sparse
     Chebyshev context encoder):
       context_gate = strength * tanh( W_z(z_basal) @ context_receiver' )
  5. z_delta MODULATES the field per gene instead of passing through an MLP
     bottleneck (the v12 lesson):
       mod_g = < W_d(g * z_delta), receiver_g >
       raw_delta += softplus(zdelta_scale) * tanh(mod)
  6. raw_delta = (softplus(gene_scale) * pair_field + bias) * (1 + context_gate)
     + self-effect scatter;  delta = max_delta * tanh(raw_delta / max_delta)

No Chebyshev anywhere: encoder side uses scFoundation tokens (v12), decoder
side has no smoothing kernel. Perturbation is single-gene (pert_idx (B,)).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v6 import WorldModelH1V6
from model_world_h1_v12 import ScFoundationEncoder, MagneticPerturbEncoder


class WorldModelH1V13(WorldModelH1V6):
    """v6 magnetic backbone + v12 scFoundation basal encoder + v12 z_delta."""

    def __init__(self, *args, scf_depth: int = 4, scf_heads: int = 8,
                 field_steps: int = 3, scf_dropout: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = self.d_model
        # --- basal branch (replaces the sparse-Chebyshev context encoder) ---
        self.scf_encoder = ScFoundationEncoder(
            self.n_genes, d_model, scf_depth, scf_heads,
            self.sparse_top_k, scf_dropout)
        self.scf_encoder.gene_emb = self.gene_emb          # share gene table
        self.zbasal_proj = nn.Linear(d_model, self.d_z)
        # --- perturbation latent branch ---
        self.pert_encoder = MagneticPerturbEncoder(
            self.n_genes, emb_dim=d_model, latent_dim=d_model, steps=field_steps)
        self.pert_encoder.gene_emb = self.gene_emb         # share gene table
        self.latent_gate = nn.Parameter(torch.tensor(-2.0))      # sigmoid ~0.12
        self.zdelta_proj = nn.Linear(d_model, self.d_z)
        self.zdelta_scale = nn.Parameter(torch.tensor(0.5406))   # softplus^-1(1)

    # -- main forward (v6 skeleton, v12 latents) ------------------------------
    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        indices = self._indices(pert_idx).to(x_ctrl.device)
        embeddings = self.gene_emb.weight

        # (3) v6 per-gene magnetic field
        magnetic_force, _, _ = self.magnet(embeddings, indices)
        source = self.source_code(embeddings[indices])
        receiver = self.receiver_code(embeddings) + self.receiver_residual
        interaction = source @ receiver.T / math.sqrt(self.d_z)

        # (1)+(2) decoupled latents
        context_input = (
            F.layer_norm(x_ctrl, (self.n_genes,))
            if self.normalize_context else x_ctrl
        )
        z_basal = self.scf_encoder(context_input)                # (B, D)
        pert_2d = indices.unsqueeze(1) if indices.dim() == 1 else indices
        z_delta = self.pert_encoder(pert_2d)                     # (B, D)

        # (4) basal state -> context gate (same semantics as v6)
        context = self.zbasal_proj(z_basal)                      # (B, d_z)
        context_receiver = F.normalize(self.context_receiver(embeddings), dim=-1)
        context_gate = self.context_strength * torch.tanh(
            context @ context_receiver.T / math.sqrt(self.d_z))

        magnetic_weight = torch.sigmoid(self.magnetic_mix)
        pair_field = magnetic_weight * magnetic_force + (1.0 - magnetic_weight) * interaction
        shared_bias_scale = self.effective_shared_bias_scale()
        raw_delta = (
            F.softplus(self.gene_scale)[None, :] * pair_field
            + shared_bias_scale * self.gene_bias[None, :]
        ) * (1.0 + context_gate)

        # (5) z_delta per-gene modulation (no MLP bottleneck)
        zd = self.zdelta_proj(torch.sigmoid(self.latent_gate) * z_delta)
        mod = zd @ receiver.T / math.sqrt(self.d_z)
        raw_delta = raw_delta + F.softplus(self.zdelta_scale) * torch.tanh(mod)

        # (6) self effect + bounded output
        self_effect = -F.softplus(self.raw_self_effect)
        raw_delta = raw_delta.scatter_add(
            1, indices[:, None], self_effect.expand(len(indices), 1))
        return self.max_delta * torch.tanh(raw_delta / self.max_delta)

    @torch.no_grad()
    def magnetic_diagnostics(self, target_genes: torch.Tensor) -> dict:
        diag = {}
        try:
            diag = super().magnetic_diagnostics(target_genes)
        except Exception:
            pass
        if target_genes.dim() == 1:
            tg = target_genes.unsqueeze(1)
        else:
            tg = target_genes
        z = self.pert_encoder(tg)
        diag.update({
            "latent_gate": float(torch.sigmoid(self.latent_gate)),
            "z_delta_norm_mean": float(z.norm(dim=-1).mean()),
            "z_delta_norm_std": float(z.norm(dim=-1).std()),
            "zdelta_scale": float(F.softplus(self.zdelta_scale)),
        })
        return diag


WorldModel = WorldModelH1V13
