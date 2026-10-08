
# -*- coding: utf-8 -*-
"""
MagWorld v16 "QuantWorld" — v13 backbone with the magnetic field replaced by
quantum Hamiltonian evolution.

What actually changes vs. the magnetic version (and what is honest physics):

  MAGNETIC v13:  field = K-step directed force rollout.  Diffusive dynamics:
                 same-sign accumulation, monotone spreading.  Response grows
                 without intrinsic bound; direction of indirect effects comes
                 almost entirely from learned per-gene scales.

  QUANTUM v16:   the perturbation injects amplitude b = |pert> at the knocked
                 gene; it evolves under a learned low-rank Hamiltonian
                     H = A_c A_o^T / sqrt(d_z)          (indefinite => ± modes)
                     field = (cos(Ht) - I) b + gamma * sin(Ht) b
                 i.e. a truncated Taylor series with ALTERNATING SIGNS.

  Genuine differences (not relabelling):
   1. INTERFERENCE: even/odd powers enter with opposite signs — excitation
      pathways cancel.  Diffusion can only spread; waves can focus.  This is
      new inductive bias for who up/down-regulates, i.e. fid.
   2. BOUNDED RESPONSE: |cos|, |sin| <= 1, so |field| <= 1 + |gamma| by
      construction — nmae-friendly, no global amplitude drift.
   3. SPECTRAL STRUCTURE: H built from the shared gene table means "who can
      talk to whom" is embedding geometry; time t is a learnable global
      propagation depth.

  What is deliberately UNCHANGED from v13 (ablation-proven): scf_encoder
  basal context (v13NoC showed removing it halves cosine), z_delta
  perturbation latent + per-gene modulation, self-effect scatter, context
  gate, tanh bounding, the source@receiver bilinear `interaction` term
  (blended with the quantum field via a learned sigmoid mix).

  Dead v6/v13 params removed: context_encoder (Chebyshev), magnet,
  magnetic_mix, reverse_mix.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v13 import WorldModelH1V13


class WorldModelH1V16(WorldModelH1V13):
    """v13 context/z_delta scaffold + quantum Hamiltonian field propagation."""

    QW_STEPS: int = 6  # Taylor order for cos(Ht)/sin(Ht)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = self.d_model
        # --- drop dead modules replaced by the quantum field ---
        if hasattr(self, "context_encoder"):
            del self.context_encoder
        if hasattr(self, "magnet"):
            del self.magnet
        for name in ("magnetic_mix", "reverse_mix"):
            if hasattr(self, name):
                delattr(self, name)
        # --- Hamiltonian generator from the shared gene table ---
        self.qw_c = nn.Linear(d_model, self.d_z, bias=False)   # column space
        self.qw_o = nn.Linear(d_model, self.d_z, bias=False)   # row space
        # evolution time (bounded to |t| <= 2 for series safety)
        self.qw_time_raw = nn.Parameter(torch.tensor(0.5))
        # odd-part (sin) mixing: gamma = 2*sigmoid(raw) - 1 in (-1, 1)
        self.qw_odd_raw = nn.Parameter(torch.tensor(0.0))
        # blend between quantum field and the bilinear interaction
        self.qw_mix = nn.Parameter(torch.tensor(0.0))

    def _happly(self, Ac: torch.Tensor, Ao: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """H v with H = A_c A_o^T / sqrt(d_z); O(G d_z), never materialized."""
        return (v @ Ao) @ Ac.T / math.sqrt(self.d_z)

    def _quantum_field(self, embeddings: torch.Tensor,
                       indices: torch.Tensor) -> torch.Tensor:
        """(cos(Ht) - I) b + gamma sin(Ht) b, b = one-hot(pert).  (B, G)"""
        G = embeddings.shape[0]
        Ac = self.qw_c(embeddings)                             # (G, d_z)
        Ao = self.qw_o(embeddings)                             # (G, d_z)
        b = F.one_hot(indices, G).float()                      # (B, G)

        t = 2.0 * torch.tanh(self.qw_time_raw)
        gamma = 2.0 * torch.sigmoid(self.qw_odd_raw) - 1.0

        # powers f_k = H^k b, k = 0..QW_STEPS
        powers = [b]
        for _ in range(self.QW_STEPS):
            powers.append(self._happly(Ac, Ao, powers[-1]))

        field = torch.zeros_like(b)
        # cos(Ht) - I: even k >= 2, signs -, +, -
        for m, k in enumerate(range(2, self.QW_STEPS + 1, 2), start=1):
            field = field + ((-1.0) ** m) * (t ** k / math.factorial(k)) * powers[k]
        # gamma * sin(Ht): odd k >= 1, signs +, -, +
        for m, k in enumerate(range(1, self.QW_STEPS + 1, 2)):
            field = field + gamma * ((-1.0) ** m) * (t ** k / math.factorial(k)) * powers[k]
        return field

    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        indices = self._indices(pert_idx).to(x_ctrl.device)
        embeddings = self.gene_emb.weight

        # (3) quantum field replaces the magnetic force rollout
        quantum_field = self._quantum_field(embeddings, indices)
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

        quantum_weight = torch.sigmoid(self.qw_mix)
        pair_field = quantum_weight * quantum_field + (1.0 - quantum_weight) * interaction
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


WorldModel = WorldModelH1V16
