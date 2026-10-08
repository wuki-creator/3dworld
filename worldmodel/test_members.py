# -*- coding: utf-8 -*-
"""Load 4 finished members (sweep-tuned calibration), solo val cosine + weight search."""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
import ensemble_predict as ep

V13A = "/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed113_np1.pt"
V13B = "/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed227_np1.pt"
M1 = "/nfs_beijing_os/zizhuo_vcc/ckpts/mccv2_h1ft_GBM8.pt"
M2 = "/nfs_beijing_os/zizhuo_vcc/ckpts/mccv2_h1ft_A172.pt"
GENES = "/home/zizhuo/vcc_data/extracted/gene_names.csv"
VAL = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz"

t0 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
genes = ep.pred3.read_genes(GENES)
gene_lookup = {g: i for i, g in enumerate(genes)}

specs = [
    (lambda: ep.MagWorldMember(V13A, device, "v13"), 0.5, 500, "v13:seed113"),
    (lambda: ep.MagWorldMember(V13B, device, "v13"), 0.5, 500, "v13:seed227"),
    (lambda: ep.MCCV2Member(M1, device), 1.0, -1, "mccv2:GBM8"),
    (lambda: ep.MCCV2Member(M2, device), 1.0, -1, "mccv2:A172"),
]
members = []
for build, scale, top_k, name in specs:
    m = build()
    assert m.genes == genes, "%s gene list mismatch" % name
    members.append((m, scale, top_k, name))
    print("loaded %-14s (%.0fs)" % (name, time.time() - t0), flush=True)

v = np.load(VAL, allow_pickle=False)
v_targets = v["targets"].astype(str)
ok = np.array([t in gene_lookup for t in v_targets])
tidx = np.asarray([gene_lookup[t] for t in v_targets[ok]], dtype=np.int64)
eff = v["effects"].astype(np.float32)[ok]
ctrl = v["controls"].astype(np.float32)[ok].mean(0)

deltas = []
for m, scale, top_k, name in members:
    raw = m.raw_delta(ctrl, tidx)
    cal = np.stack([ep.calibrate(row, ti, top_k, scale, 1.0, 5.0) for row, ti in zip(raw, tidx)])
    deltas.append(cal)
    print("solo %-14s val cosine=%.4f" % (name, ep.cosine_row(cal, eff).mean()), flush=True)

stacked = np.stack(deltas)
print("equal-weight ensemble val cosine=%.4f" % ep.cosine_row(stacked.mean(0), eff).mean(), flush=True)
rngw = np.random.default_rng(7)
best_score = -1.0
best_w = None
for _ in range(4000):
    w = rngw.dirichlet(np.ones(len(members)))
    s = float(ep.cosine_row(np.tensordot(w, stacked, axes=1), eff).mean())
    if s > best_score:
        best_score = s
        best_w = w
print("best weights %s val cosine=%.4f" % (np.round(best_w, 3).tolist(), best_score), flush=True)
print("TEST_DONE (%.0fs)" % (time.time() - t0), flush=True)
