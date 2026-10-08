# -*- coding: utf-8 -*-
"""MagWorld v25 = v18 + 异构图先验 GNN（静态低自由度注入，无 per-sample 注意力）.

v24 的教训：kq 注意力 aff∈(0,1)^{B×G} 是 per-(样本,基因) 的自由度，
60 epochs 即可记住训练集每个扰动的特异性 → 训练集 raw_cosine 0.40+
但 val 仅 0.374（过拟合 gap 0.03）。异构图编码器本身没学动。

v25 保留异构图 GNN 先验编码器（u = LayerNorm(W_self h + Σ_r β_r m^r)），
但先验只走三条 O(G) 静态通道（全部零初始化 → 初始函数 ≡ v18）：

  1. 磁矩强度:   charge_i = softplus(charge(e_i) + charge_bias + w_c·u_i)
  2. 响应易感度: pre_a_i += w_a·u_i        （v18 dgidb amp hook 的图学习版）
  3. 源增益:     src_gain_t = (1 + tanh(w_s·f_t)) · (1 + tanh(w_src·u_t))

没有 per-sample × per-gene 的记忆面，先验只能调制基因静态属性。
世界模型训练法不变（train_v14_aux.py）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v18 import WorldModelH1V18, _slog
from model_world_h1_v24 import _group_softmax


class WorldModelH1V25(WorldModelH1V18):
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
        d = self.d_z

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

        # ---- static low-DoF prior injection (zero-init ≡ v18 at start) ----
        self.prior_charge = nn.Linear(d, 1, bias=False)
        self.prior_amp = nn.Linear(d, 1, bias=False)
        self.prior_src = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.prior_charge.weight)
        nn.init.zeros_(self.prior_amp.weight)
        nn.init.zeros_(self.prior_src.weight)

    # ---------------------------------------------------------- prior encoder
    def _prior_encode(self, e: torch.Tensor) -> torch.Tensor:
        h = self.prior_self(e)
        msgs = []
        for r in range(self.n_relations):
            ei = getattr(self, "rel%d_ei" % r)
            es = getattr(self, "rel%d_es" % r)
            ew = getattr(self, "rel%d_ew" % r)
            src, dst = ei[0], ei[1]
            wh = self.rel_proj[r](e)
            ws, wd = wh[src], wh[dst]
            logit = self.rel_att[r](torch.cat([ws, wd], -1)).squeeze(-1)
            alpha = _group_softmax(logit, dst, e.shape[0])
            m = torch.zeros_like(h)
            m.index_add_(0, dst, (alpha * es * ew).unsqueeze(-1) * ws)
            msgs.append(m)
        beta = torch.softmax(self.rel_gate, dim=0)
        agg = (beta.view(-1, 1, 1) * torch.stack(msgs, 0)).sum(0)
        return self.prior_norm(h + agg)

    # ------------------------------------------------------------- main path
    def predict_delta(self, x_ctrl: torch.Tensor, pert_idx) -> torch.Tensor:
        if self.n_relations == 0:
            return super().predict_delta(x_ctrl, pert_idx)
        device = x_ctrl.device
        indices = self._indices(pert_idx).to(device)
        B = x_ctrl.shape[0]
        e = self.gene_emb.weight
        G = e.shape[0]

        u = self._prior_encode(e)                                   # (G, d_z)
        # static prior channels
        charge_prior = self.prior_charge(u).squeeze(-1)             # (G,)
        amp_prior = self.prior_amp(u).squeeze(-1)                   # (G,)
        src_prior = 1.0 + torch.tanh(
            self.prior_src(u[indices]).squeeze(-1)
        )                                                           # (B,)

        pos = F.normalize(self.pole(e), dim=-1)
        charge = F.softplus(
            self.charge(e).squeeze(-1) + self.charge_bias + charge_prior
        )

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

        # ---- v18 DGIDB hooks + static prior src channel ----
        f = getattr(self, "dgidb_feat", None)
        if f is not None:
            f_src = f[indices]
            src_gain = (1.0 + torch.tanh(self.src_proj(f_src)).squeeze(-1)) * src_prior
            amp_bias = self.amp_dg(f).squeeze(-1) + amp_prior
        else:
            src_gain = src_prior
            amp_bias = amp_prior

        phi = kappa * charge[indices][:, None] * k_src * src_gain[:, None]
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
            diag["prior_charge_w"] = float(self.prior_charge.weight.abs().sum())
            diag["prior_amp_w"] = float(self.prior_amp.weight.abs().sum())
            diag["prior_src_w"] = float(self.prior_src.weight.abs().sum())
        return diag


WorldModel = WorldModelH1V25
