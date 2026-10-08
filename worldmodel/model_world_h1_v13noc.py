
# -*- coding: utf-8 -*-
"""
MagWorld v13-NoC — v13 with the graph-encoding context branch removed.

Background: in v13 the inherited v6 context_encoder (Chebyshev graph conv +
value_projection) is DEAD CODE — predict_delta never calls it; its tensors
only sit in the state_dict because v6's __init__ registers them.  The LIVE
context pathway in v13 is:

    z_basal = ScFoundationEncoder(LN(x_ctrl))   # gene-token attention (graph)
    context = zbasal_proj(z_basal)
    context_gate = strength * tanh(context @ context_receiver')

This ablation therefore does two things:
  1. deletes the dead v6 context_encoder entirely (Chebyshev + value proj);
  2. replaces the scf_encoder attention with a single linear projection of
     the (layer-normalized) control expression:
         z_basal = W @ context_input          # (B, d_model)

Everything else — magnetic field rollout, interaction bilinear, z_delta
per-gene modulation, self-effect scatter, tanh bounding — is byte-identical
to v13.  If val cosine/fid/sign_acc barely move vs v13, the attention-based
basal encoding was not carrying weight; if they drop, it matters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v13 import WorldModelH1V13


class WorldModelH1V13NoC(WorldModelH1V13):
    """v13 minus the graph-encoding context branch (linear basal instead)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = self.d_model
        # (1) drop the dead v6 Chebyshev context encoder (never called in v13)
        if hasattr(self, "context_encoder"):
            del self.context_encoder
        # (2) replace scFoundation attention with a plain linear projection
        self.lin_basal = nn.Linear(self.n_genes, d_model)
        del self.scf_encoder

    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        indices = self._indices(pert_idx).to(x_ctrl.device)
        embeddings = self.gene_emb.weight

        # (3) v6 per-gene magnetic field (unchanged)
        magnetic_force, _, _ = self.magnet(embeddings, indices)
        source = self.source_code(embeddings[indices])
        receiver = self.receiver_code(embeddings) + self.receiver_residual
        interaction = source @ receiver.T / math.sqrt(self.d_z)

        # (1) linear basal state — no graph encoding
        context_input = (
            F.layer_norm(x_ctrl, (self.n_genes,))
            if self.normalize_context else x_ctrl
        )
        z_basal = self.lin_basal(context_input)                  # (B, D)
        pert_2d = indices.unsqueeze(1) if indices.dim() == 1 else indices
        z_delta = self.pert_encoder(pert_2d)                     # (B, D)

        # (4) basal state -> context gate (same semantics as v13)
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

        # (5) z_delta per-gene modulation (unchanged)
        zd = self.zdelta_proj(torch.sigmoid(self.latent_gate) * z_delta)
        mod = zd @ receiver.T / math.sqrt(self.d_z)
        raw_delta = raw_delta + F.softplus(self.zdelta_scale) * torch.tanh(mod)

        # (6) self effect + bounded output (unchanged)
        self_effect = -F.softplus(self.raw_self_effect)
        raw_delta = raw_delta.scatter_add(
            1, indices[:, None], self_effect.expand(len(indices), 1))
        return self.max_delta * torch.tanh(raw_delta / self.max_delta)


WorldModel = WorldModelH1V13NoC
