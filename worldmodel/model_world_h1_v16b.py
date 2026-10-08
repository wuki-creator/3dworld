
# -*- coding: utf-8 -*-
"""
MagWorld v16b — v16 "QuantWorld" with spectral-normalized Hamiltonian.

v16 bug: the truncated Taylor series for cos(Ht)/sin(Ht) only converges for
|lambda_max(H) * t| << K.  Training drove H's eigenvalues to ~ +/-14 while
t -> 0.96, so |lambda t| ~ 13 >> 6 and the series diverged: the "quantum
field" was an uncontrolled degree-6 polynomial (field max 2.66 exceeded the
theoretical bound 2).  Diagnosis: qw_mix stayed 0.5 and the branch trained,
but it was not Hamiltonian evolution.

Fix: normalize H to unit spectral radius every forward pass.  H and the
dz x dz matrix M = A_o^T A_c / sqrt(d_z) share nonzero eigenvalues, so
sn = sigma_max(M) is computed exactly (dz=64, cheap, differentiable).
H <- H / sn then makes t a genuine evolution time bounded by 2, the K=6
series converges (tail <= 2^6/720 ~ 0.09), |field| <= 1 + |gamma| holds,
and the odd (sin / directed-propagation) part can actually engage.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from model_world_h1_v16 import WorldModelH1V16


class WorldModelH1V16B(WorldModelH1V16):
    """v16 with spectrally normalized H (valid Taylor series)."""

    def _quantum_field(self, embeddings: torch.Tensor,
                       indices: torch.Tensor) -> torch.Tensor:
        """(cos(Ht) - I) b + gamma sin(Ht) b with H spectrally normalized."""
        G = embeddings.shape[0]
        Ac = self.qw_c(embeddings)                             # (G, d_z)
        Ao = self.qw_o(embeddings)                             # (G, d_z)
        b = F.one_hot(indices, G).float()                      # (B, G)

        # exact spectral norm of H via the dz x dz companion matrix
        M = (Ao.T @ Ac) / math.sqrt(self.d_z)                  # (d_z, d_z)
        sn = torch.linalg.matrix_norm(M, ord=2).clamp(min=1e-6)

        t = 2.0 * torch.tanh(self.qw_time_raw)
        gamma = 2.0 * torch.sigmoid(self.qw_odd_raw) - 1.0

        def hstep(v: torch.Tensor) -> torch.Tensor:
            return ((v @ Ao) @ Ac.T) / (math.sqrt(self.d_z) * sn)

        powers = [b]
        for _ in range(self.QW_STEPS):
            powers.append(hstep(powers[-1]))

        field = torch.zeros_like(b)
        for m, k in enumerate(range(2, self.QW_STEPS + 1, 2), start=1):
            field = field + ((-1.0) ** m) * (t ** k / math.factorial(k)) * powers[k]
        for m, k in enumerate(range(1, self.QW_STEPS + 1, 2)):
            field = field + gamma * ((-1.0) ** m) * (t ** k / math.factorial(k)) * powers[k]
        return field


WorldModel = WorldModelH1V16B
