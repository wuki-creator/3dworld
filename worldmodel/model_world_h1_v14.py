# -*- coding: utf-8 -*-
"""
MagWorld v14 — physics-guided magnetic-flow world model (P0/P1/P2 upgrades).

Gene-level magnetic graph (nodes = genes, kNN in embedding space):
  * Learnable radial kernel profile  K(r) = sum_l alpha_l * r^{-l},  l = 1..4
    with L2 pull toward the physics init [1, 0, 1/3, 0] (monopole:dipole = 1:1/3).
  * Baseline magnetic moment per gene from the *control* expression
    (paramagnetic ground state):  m_i = q_i * (1 + tanh(w ln(1+x_i) + b)).
    The perturbation source is injected explicitly:  + kappa * q_s * K(r_is).
  * Scalar potential phi (amplitude carrier) and vector field V = K * m * r_hat
    (direction carrier) are computed on sparse kNN edges only: O(G*k).
  * Langevin saturation (P2-9):  V <- H_max * L(|V|/H_max) * V/|V|,
    L(a) = coth(a) - 1/a,  H_max per-gene learnable (expression ceiling).
  * Amplitude/direction factorization (P0-1):
        a_i = sigmoid(w . [s_i, phi_i, |V_i|, H_ext])
        d_i = normalize(d0_i + eta * (V_i - (V_i . d0_i) d0_i)),  d0 = normalize(W_d s_i)
        gate_i = sigmoid(w_g . d_i)
  * Directed response matrix (P0-3):  r = phi @ S2 @ S1^T / sqrt(rank),
    S1 != S2  ->  d(dx_i)/dB_j != d(dx_j)/dB_i  (no symmetry constraint).
  * Residual field layers (P0-2, <= 3):  h <- h + a * tanh(r) * gate,
    moments relax:  m <- m + rho * dx.
  * Base + residual split (P1-7): a validated direct base pathway
    (shared bias + context gate) predicts the bulk response; the magnetic
    module only predicts the residual correction.

Loss (trainer side, P0-4):  L1 + lambda_mmd * MMD(pred, truth)
  + lambda_bce * BCE(sign) + lambda_kernel * ||alpha - alpha_init||^2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(x: float) -> float:
    return math.log(math.expm1(x))


def _slog(x: torch.Tensor) -> torch.Tensor:
    """Signed log1p: compresses heavy-tailed field magnitudes."""
    return torch.sign(x) * torch.log1p(x.abs())


class WorldModelH1V14(nn.Module):
    """Magnetic-flow world model with learnable physics-guided kernel."""

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        d_z: int = 64,
        d_hidden: int = 192,
        d_dir: int = 32,
        max_delta: float = 1.5,
        context_strength: float = 0.25,
        shared_bias_mode: str = "gated",
        shared_bias_initial_scale: float = 0.1,
        normalize_context: bool = True,
        graph_k: int = 50,
        n_layers: int = 3,
        response_rank: int = 64,
        use_dipole: bool = True,
        use_langevin: bool = True,
        langevin_init: float = 4.0,
        kernel_powers: int = 4,
    ) -> None:
        super().__init__()
        if shared_bias_mode not in {"free", "gated", "none"}:
            raise ValueError("invalid shared_bias_mode")
        self.n_genes = n_genes
        self.d_model = d_model
        self.d_z = d_z
        self.d_dir = d_dir
        self.max_delta = max_delta
        self.context_strength = context_strength
        self.shared_bias_mode = shared_bias_mode
        self.normalize_context = normalize_context
        self.graph_k = min(graph_k, n_genes - 1)
        self.n_layers = max(1, min(n_layers, 3))
        self.response_rank = response_rank
        self.use_dipole = use_dipole
        self.use_langevin = use_langevin

        # ---- frozen gene embeddings (hybrid scGPT + coexpression) ----
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.gene_emb.weight.requires_grad_(False)

        # ---- pole positions and charges ----
        self.pole = nn.Linear(d_model, d_z, bias=False)
        self.charge = nn.Linear(d_model, 1)
        self.charge_bias = nn.Parameter(torch.tensor(-0.5))

        # ---- learnable radial kernel profile, physics init [1, 0, 1/3, 0] ----
        self.kernel_powers = kernel_powers
        init = torch.zeros(kernel_powers)
        init[0] = 1.0                      # monopole 1/r (Poisson Green)
        if kernel_powers > 2 and use_dipole:
            init[2] = 1.0 / 3.0            # dipole 1/r^3
        self.raw_kernel_alpha = nn.Parameter(
            torch.tensor([_softplus_inverse(v) if v > 0 else -4.0 for v in init])
        )
        self.register_buffer(
            "kernel_alpha_init", F.softplus(self.raw_kernel_alpha).detach().clone()
        )

        # explicit perturbation-source injection strength
        self.raw_source_kappa = nn.Parameter(torch.tensor(_softplus_inverse(1.0)))

        # ---- baseline moment from control expression (paramagnetic ground) ----
        self.moment_expr_w = nn.Parameter(torch.tensor(1.0))
        self.moment_expr_b = nn.Parameter(torch.tensor(0.0))

        # ---- context (basal state) encoder: control expression -> global latent ----
        self.context_encoder = nn.Sequential(
            nn.Linear(n_genes, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_z),
            nn.LayerNorm(d_z),
        )
        # per-gene state s_i = W_e(e_i) + W_z(z)
        self.state_gene = nn.Linear(d_model, d_dir, bias=False)
        self.state_ctx = nn.Linear(d_z, d_dir, bias=False)

        # ---- amplitude channel: a_i = sigmoid(w . [s_i, phi_i, |V_i|, H_ext]) ----
        self.amp_gene = nn.Linear(d_dir, 1, bias=False)
        self.amp_phi = nn.Linear(1, 1, bias=False)
        self.amp_v = nn.Linear(1, 1, bias=False)
        self.amp_ext = nn.Linear(1, 1, bias=False)
        self.amp_bias = nn.Parameter(torch.tensor(2.0))

        # ---- direction channel ----
        self.dir_proj = nn.Linear(d_dir, d_z, bias=False)
        self.raw_eta = nn.Parameter(torch.tensor(0.0))       # orthogonal correction
        self.gate_proj = nn.Linear(d_z, 1, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

        # ---- directed response matrix (asymmetric low-rank) ----
        # r = phi @ S2 @ S1^T / sqrt(rank); S1 != S2 (no symmetry constraint)
        self.response_left = nn.Linear(n_genes, response_rank, bias=False)   # weight (rank, G)
        self.response_right = nn.Linear(n_genes, response_rank, bias=False)  # weight (rank, G)

        # ---- Langevin saturation ceiling (per gene) ----
        self.raw_h_max = nn.Parameter(torch.full((n_genes,), _softplus_inverse(langevin_init)))

        # ---- moment relaxation (residual layers) ----
        self.raw_relax = nn.Parameter(torch.tensor(_softplus_inverse(0.5)))

        # ---- base pathway (validated v4 semantics) ----
        self.context_receiver = nn.Linear(d_model, d_z, bias=False)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        initial_logit = math.log(shared_bias_initial_scale / (1.0 - shared_bias_initial_scale))
        self.raw_shared_bias_scale = nn.Parameter(torch.tensor(initial_logit))
        self.raw_self_effect = nn.Parameter(torch.tensor(1.9))  # softplus(1.9) ~ 2.0 knockdown

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def initialize_gene_embeddings(self, features: torch.Tensor) -> None:
        if features.shape != self.gene_emb.weight.shape:
            raise ValueError(
                "expected gene features %s, got %s"
                % (tuple(self.gene_emb.weight.shape), tuple(features.shape))
            )
        self.gene_emb.weight.copy_(F.normalize(features.float(), dim=-1))

    @staticmethod
    def _indices(pert_idx) -> torch.Tensor:
        if isinstance(pert_idx, torch.Tensor):
            return pert_idx.reshape(-1).long()
        return torch.stack([value.reshape(-1)[0] for value in pert_idx]).long()

    def effective_shared_bias_scale(self) -> torch.Tensor:
        if self.shared_bias_mode == "free":
            return self.gene_bias.new_tensor(1.0)
        if self.shared_bias_mode == "gated":
            return torch.sigmoid(self.raw_shared_bias_scale)
        return self.gene_bias.new_tensor(0.0)

    def kernel_alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_kernel_alpha)

    @staticmethod
    def _langevin(a: torch.Tensor) -> torch.Tensor:
        """L(a) = coth(a) - 1/a, numerically stable for small a."""
        small = a.abs() < 1e-3
        a_safe = a.clamp_min(1e-3)
        out = 1.0 / torch.tanh(a_safe) - 1.0 / a_safe
        return torch.where(small, a / 3.0, out)

    def _knn_edges(self, pos: torch.Tensor) -> torch.Tensor:
        """Top-k neighbors by pole alignment. pos: (G, d_z) normalized."""
        sim = pos @ pos.T
        _, idx = torch.topk(sim, k=self.graph_k + 1, dim=1)
        return idx[:, 1:]  # drop self

    def _kernel(self, r: torch.Tensor) -> torch.Tensor:
        """K(r) = sum_l alpha_l r^{-l}. r: positive distances, any shape."""
        r = r.clamp_min(1e-4)
        alpha = self.kernel_alpha()
        k = alpha[0] / r
        for l in range(1, self.kernel_powers):
            k = k + alpha[l] * torch.pow(r, -(l + 1))
        return k

    # -------------------------------------------------------------- main path
    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        device = x_ctrl.device
        B = x_ctrl.shape[0]
        indices = self._indices(pert_idx).to(device)
        e = self.gene_emb.weight                                  # (G, d_model)
        G = e.shape[0]

        # positions / charges
        pos = F.normalize(self.pole(e), dim=-1)                   # (G, d_z)
        charge = F.softplus(self.charge(e).squeeze(-1) + self.charge_bias)  # (G,)

        # sparse kNN edges in pole space (indices constant per step)
        nb = self._knn_edges(pos.detach())                        # (G, k)
        p_nb = pos[nb]                                            # (G, k, d_z)
        diff = pos[:, None, :] - p_nb                             # (G, k, d_z)
        r_nb = diff.norm(dim=-1).clamp_min(1e-4)                  # (G, k)
        k_nb = self._kernel(r_nb)                                 # (G, k)
        k_nb = k_nb / (k_nb.sum(-1, keepdim=True) + 1e-6)  # attention-like weights
        uvec = diff / r_nb.unsqueeze(-1)                          # unit j -> i

        # ---- context (basal) state ----
        ctx_in = F.layer_norm(x_ctrl, (G,)) if self.normalize_context else x_ctrl
        z = self.context_encoder(ctx_in)                          # (B, d_z)
        s = self.state_gene(e)[None, :, :] + self.state_ctx(z)[:, None, :]  # (B,G,d_dir)

        # ---- baseline moments from control expression ----
        expr = torch.log1p(x_ctrl.clamp_min(0.0))                 # (B, G)
        moment = charge[None, :] * (
            1.0 + torch.tanh(self.moment_expr_w * expr + self.moment_expr_b)
        )                                                         # (B, G)

        # ---- explicit source injection ----
        kappa = F.softplus(self.raw_source_kappa)
        p_src = pos[indices]                                      # (B, d_z)
        src_diff = pos[None, :, :] - p_src[:, None, :]            # (B, G, d_z)
        d_src = src_diff.norm(dim=-1).clamp_min(1e-4)             # (B, G)
        k_src = self._kernel(d_src)                               # (B, G)
        k_src = k_src / (k_src.mean(dim=1, keepdim=True) + 1e-6)  # bound heavy tail
        u_src = src_diff / d_src.unsqueeze(-1)
        phi = kappa * charge[indices][:, None] * k_src            # (B, G)
        vec = (kappa * charge[indices][:, None] * k_src)[:, :, None] * u_src  # (B,G,d_z)

        # ---- residual field layers ----
        h = torch.zeros(B, G, device=device)
        relax = F.softplus(self.raw_relax)
        h_max = F.softplus(self.raw_h_max)[None, :]               # (1, G)
        for _ in range(self.n_layers):
            # sparse message pass: scalar potential and vector field
            m_nb = moment[:, nb.reshape(-1)].reshape(B, G, nb.shape[1])  # (B,G,k)
            phi = phi + (k_nb[None] * m_nb).sum(-1)               # (B, G)
            vec = vec + ((k_nb[None] * m_nb).unsqueeze(-1) * uvec[None]).sum(2)
            # Langevin saturation on the vector field
            v_norm = vec.norm(dim=-1).clamp_min(1e-6)             # (B, G)
            if self.use_langevin:
                scale = h_max * self._langevin(v_norm / h_max) / v_norm
            else:
                scale = 1.0 / v_norm
            vec_s = vec * scale.unsqueeze(-1)                     # |V| <= H_max

            # amplitude channel
            h_ext = v_norm.mean(1, keepdim=True)                  # (B, 1) global field
            pre_a = (
                self.amp_gene(s).squeeze(-1)
                + self.amp_phi(_slog(phi).unsqueeze(-1)).squeeze(-1)
                + self.amp_v(_slog(v_norm).unsqueeze(-1)).squeeze(-1)
                + self.amp_ext(h_ext)
                + self.amp_bias
            )
            if getattr(self, "_debug", False):
                self._dbg_pre_a = dict(
                    min=float(pre_a.min()), mean=float(pre_a.mean()),
                    max=float(pre_a.max()),
                    g=float(self.amp_gene.weight.abs().max()),
                    p=float(self.amp_phi.weight.abs().max()),
                    v=float(self.amp_v.weight.abs().max()),
                    x=float(self.amp_ext.weight.abs().max()),
                )
            a = torch.sigmoid(pre_a)                              # (B, G)

            # direction channel with orthogonal correction
            d0 = F.normalize(self.dir_proj(s), dim=-1)            # (B, G, d_dir)
            vproj = vec_s - (vec_s * d0).sum(-1, keepdim=True) * d0
            eta = torch.sigmoid(self.raw_eta) * 2.0
            d = F.normalize(d0 + eta * vproj, dim=-1)
            gate = torch.sigmoid(self.gate_proj(d).squeeze(-1) + self.gate_bias)

            # directed response matrix on RMS-normalized potential
            # (keeps r at O(1) so tanh(r) stays in its active region)
            phi_n = _slog(phi)
            phi_n = phi_n / phi_n.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            r = self.response_right(phi_n) @ self.response_left.weight  # (B, G)

            dx = a * torch.tanh(r) * gate                         # (B, G)
            h = h + dx
            moment = moment + relax * dx                          # moment relaxation

        # ---- base pathway (bulk) + magnetic residual ----
        if getattr(self, "_debug", False):
            self._dbg = dict(
                h=float(h.abs().max()), phi=float(phi.abs().max()),
                a=float(a.mean()), gate=float(gate.mean()),
                r=float(r.abs().max()), vnorm=float(v_norm.max()),
                dx=float(dx.abs().max()),
            )
        ctx_rec = F.normalize(self.context_receiver(e), dim=-1)
        ctx_gate = self.context_strength * torch.tanh(
            z @ ctx_rec.T / math.sqrt(self.d_z)
        )                                                         # (B, G)
        shared = self.effective_shared_bias_scale()
        base = shared * self.gene_bias[None, :] * (1.0 + ctx_gate)
        raw = base + h

        # self effect: knockdown reduces the target gene itself
        self_effect = -F.softplus(self.raw_self_effect)
        raw = raw.scatter_add(
            1, indices[:, None], self_effect.expand(len(indices), 1)
        )
        return self.max_delta * torch.tanh(raw / self.max_delta)

    def forward(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        return x_ctrl + self.predict_delta(x_ctrl, pert_idx)

    @torch.no_grad()
    def magnetic_diagnostics(self, pert_idx) -> dict:
        idx = self._indices(pert_idx).to(self.gene_emb.weight.device)
        e = self.gene_emb.weight
        pos = F.normalize(self.pole(e), dim=-1)
        charge = F.softplus(self.charge(e).squeeze(-1) + self.charge_bias)
        nb = self._knn_edges(pos)
        r_nb = (pos[:, None, :] - pos[nb]).norm(dim=-1)
        alpha = self.kernel_alpha()
        return {
            "kernel_alpha": [round(float(v), 4) for v in alpha],
            "kernel_alpha_init": [round(float(v), 4) for v in self.kernel_alpha_init],
            "source_kappa": float(F.softplus(self.raw_source_kappa)),
            "charge_mean": float(charge.mean()),
            "charge_abs_max": float(charge.abs().max()),
            "neighbor_distance_mean": float(r_nb.mean()),
            "eta": float(torch.sigmoid(self.raw_eta) * 2.0),
            "relax": float(F.softplus(self.raw_relax)),
            "h_max_mean": float(F.softplus(self.raw_h_max).mean()),
            "shared_bias_scale": float(self.effective_shared_bias_scale()),
            "self_effect": float(-F.softplus(self.raw_self_effect)),
            "use_dipole": bool(self.use_dipole),
            "use_langevin": bool(self.use_langevin),
            "n_layers": int(self.n_layers),
            "pert_sample": idx[:8].tolist(),
        }


WorldModel = WorldModelH1V14
