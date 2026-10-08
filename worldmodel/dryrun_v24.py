# -*- coding: utf-8 -*-
"""Dry-run v24: forward/backward, grad flow, strict state-dict round-trip."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
from model_world_h1_v24 import WorldModelH1V24  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
G, DM, DZ = 18533, 256, 64
cfg = dict(
    n_genes=G, d_model=DM, d_z=DZ, d_hidden=192, d_dir=32,
    max_delta=1.5, context_strength=0.25, graph_k=50, n_layers=3,
    response_rank=64, kernel_powers=4, n_dgidb_feats=8,
    prior_graph_paths=[
        "/nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_graph.npz",
        "/nfs_beijing_os/zizhuo_vcc/embeddings/trrust_graph.npz",
    ],
)
m = WorldModelH1V24(**cfg).cuda()
m.initialize_gene_embeddings(torch.randn(G, DM).cuda())
m.load_dgidb_prior(np.random.rand(G, 8).astype(np.float32))

x = torch.rand(4, G).cuda() * 5.0
idx = torch.randint(0, G, (4,)).cuda()
out = m(x, idx)
assert out.shape == (4, G), out.shape
loss = m.predict_delta(x, idx).pow(2).mean()
loss.backward()
params = dict(m.named_parameters())
need = ["rel_gate", "kq_key.weight", "kq_query.weight", "kq_readout.weight",
        "raw_gamma", "raw_lam", "prior_self.weight",
        "rel_proj.0.weight", "rel_proj.1.weight",
        "rel_att.0.weight", "rel_att.1.weight"]
for n in need:
    g = params[n].grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n
print("GRAD_OK", {n: round(float(params[n].grad.abs().sum()), 6) for n in need})

# strict state-dict round trip (proxy_score style)
sd = m.state_dict()
m2 = WorldModelH1V24(**cfg).cuda()
m2.load_state_dict(sd)  # strict
print("STRICT_LOAD_OK")

# near-v18 equivalence at init: gamma/lam ~ 0.018
print("gamma", float(torch.nn.functional.softplus(m.raw_gamma)),
      "lam", float(torch.nn.functional.softplus(m.raw_lam)))
with torch.no_grad():
    d = m.magnetic_diagnostics(idx)
    print("diag_relations", d.get("relation_names"), d.get("relation_beta"))
print("DRYRUN_ALL_OK")
