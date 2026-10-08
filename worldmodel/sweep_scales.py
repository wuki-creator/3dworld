# -*- coding: utf-8 -*-
"""Per-member scale sweep on H1 validation signatures."""
import sys

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

device = torch.device("cuda")
genes = ep.pred3.read_genes(GENES)
gene_lookup = {g: i for i, g in enumerate(genes)}

v = np.load(VAL, allow_pickle=False)
v_targets = v["targets"].astype(str)
ok = np.array([t in gene_lookup for t in v_targets])
tidx = np.asarray([gene_lookup[t] for t in v_targets[ok]], dtype=np.int64)
eff = v["effects"].astype(np.float32)[ok]
ctrl = v["controls"].astype(np.float32)[ok].mean(0)

specs = [
    (ep.MagWorldMember(V13A, device, "v13"), "v13:seed113", [0.1, 0.25, 0.5, 1.0]),
    (ep.MagWorldMember(V13B, device, "v13"), "v13:seed227", [0.1, 0.25, 0.5, 1.0]),
    (ep.MCCV2Member(M1, device), "mccv2:GBM8", [0.5, 1.0, 2.0, 4.0]),
    (ep.MCCV2Member(M2, device), "mccv2:A172", [0.5, 1.0, 2.0, 4.0]),
]
for m, name, scales in specs:
    raw = m.raw_delta(ctrl, tidx)
    print("%s raw cosine=%.4f" % (name, ep.cosine_row(raw, eff).mean()), flush=True)
    for s in scales:
        cal = np.stack([ep.calibrate(row, ti, 500, s, 1.0, 5.0) for row, ti in zip(raw, tidx)])
        print("  scale=%.2f -> %.4f" % (s, ep.cosine_row(cal, eff).mean()), flush=True)
print("SWEEP_DONE", flush=True)
