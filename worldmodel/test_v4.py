# -*- coding: utf-8 -*-
"""5th member (v4 seed337) check: load, scale sweep, 5-member weight search."""
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
V4 = "/nfs_beijing_os/zizhuo_vcc/ckpts/v4_seed337"
GENES = "/home/zizhuo/vcc_data/extracted/gene_names.csv"
VAL = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz"

t0 = time.time()
device = torch.device("cuda")
genes = ep.pred3.read_genes(GENES)
gene_lookup = {g: i for i, g in enumerate(genes)}

v4 = ep.MagWorldMember(V4, device, "v4")
assert v4.genes == genes, "v4 gene list mismatch"
print("loaded v4:seed337 arch=%s (%.0fs)" % (v4.kind, time.time() - t0), flush=True)

v = np.load(VAL, allow_pickle=False)
v_targets = v["targets"].astype(str)
ok = np.array([t in gene_lookup for t in v_targets])
tidx = np.asarray([gene_lookup[t] for t in v_targets[ok]], dtype=np.int64)
eff = v["effects"].astype(np.float32)[ok]
ctrl = v["controls"].astype(np.float32)[ok].mean(0)

raw4 = v4.raw_delta(ctrl, tidx)
print("v4 raw val cosine=%.4f" % ep.cosine_row(raw4, eff).mean(), flush=True)
best_s, best_c = None, -1
for s in (0.5, 1.0, 2.0, 4.0):
    cal = np.stack([ep.calibrate(row, ti, 500, s, 1.0, 5.0) for row, ti in zip(raw4, tidx)])
    c = ep.cosine_row(cal, eff).mean()
    print("  v4 scale=%.2f -> %.4f" % (s, c), flush=True)
    if c > best_c:
        best_c, best_s = c, s
print("v4 chosen scale=%.2f (%.4f)" % (best_s, best_c), flush=True)

specs = [
    (ep.MagWorldMember(V13A, device, "v13"), 0.5, 500, "v13:seed113"),
    (ep.MagWorldMember(V13B, device, "v13"), 0.5, 500, "v13:seed227"),
    (ep.MCCV2Member(M1, device), 1.0, -1, "mccv2:GBM8"),
    (ep.MCCV2Member(M2, device), 1.0, -1, "mccv2:A172"),
]
deltas = []
names = []
for m, scale, top_k, name in specs:
    raw = m.raw_delta(ctrl, tidx)
    deltas.append(np.stack([ep.calibrate(row, ti, top_k, scale, 1.0, 5.0) for row, ti in zip(raw, tidx)]))
    names.append(name)
deltas.append(np.stack([ep.calibrate(row, ti, 500, best_s, 1.0, 5.0) for row, ti in zip(raw4, tidx)]))
names.append("v4:seed337")

stacked = np.stack(deltas)
for n, d in zip(names, deltas):
    print("solo %-14s val cosine=%.4f" % (n, ep.cosine_row(d, eff).mean()), flush=True)
print("equal-weight 5-member=%.4f" % ep.cosine_row(stacked.mean(0), eff).mean(), flush=True)
rngw = np.random.default_rng(7)
best_score, best_w = -1.0, None
for _ in range(8000):
    w = rngw.dirichlet(np.ones(len(deltas)))
    s = float(ep.cosine_row(np.tensordot(w, stacked, axes=1), eff).mean())
    if s > best_score:
        best_score, best_w = s, w
print("best weights %s val cosine=%.4f" % (np.round(best_w, 3).tolist(), best_score), flush=True)
print("V4TEST_DONE (%.0fs)" % (time.time() - t0), flush=True)
