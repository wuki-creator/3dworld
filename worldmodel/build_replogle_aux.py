# -*- coding: utf-8 -*-
"""Build Replogle auxiliary signatures from Harmonizome standardized matrices.
Output: /nfs_beijing_os/zizhuo_vcc/pertseq/replogle_aux.npz
  genes    (G,) str    panel gene symbols (order)
  targets  (M,) str    perturbed gene symbol per condition
  ctrl     (G,) float32 zeros (x_ctrl supplied by Norman ctrl at train time if available)
  effects  (M, G) float16 standardized signature (full panel width)
  cells    (M,) float32 synthetic counts
"""
import csv
import gzip
import numpy as np

BASE = "/nfs_beijing_os/zizhuo_vcc"
SIG = f"{BASE}/signatures/h1_trainval_signatures.npz"
panel = np.load(SIG)["genes"].astype(str).tolist()
pidx = {g: i for i, g in enumerate(panel)}
G = len(panel)

import json as _json
with gzip.open(f"{BASE}/pertseq/canon_map.json.gz", "rt", encoding="utf-8") as fh:
    _canon_map = _json.load(fh)
def canon(g):
    return _canon_map.get(g.strip().upper(), g.strip().upper())

FILES = ["reploglek562genomewide", "reploglek562essential", "reploglerpe1essential"]
targets_all = []
blocks = []
for name in FILES:
    path = f"{BASE}/pertseq/replogle/{name}_std.txt.gz"
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        attrs = []
        for a in header[1:]:
            p = a.split("_")
            if len(p) < 2:
                continue
            sym = p[1].strip()
            if not sym or "NON-TARGET" in sym.upper():
                continue
            attrs.append(canon(sym))
        acc = np.zeros((G, len(attrs)), dtype=np.float32)
        cnt = np.zeros(G, dtype=np.float32)
        n = 0
        for line in fh:
            row = line.rstrip("\n").split("\t")
            gi = pidx.get(canon(row[0]), -1)
            if gi < 0:
                continue
            v = np.array([float(x) if x not in ("", "NA", "NaN") else 0.0
                          for x in row[1:1 + len(attrs)]], dtype=np.float32)
            acc[gi] += v
            cnt[gi] += 1.0
            n += 1
        acc /= np.maximum(cnt[:, None], 1.0)
    eff = acc.T  # (attrs, G)
    eff = eff - eff.mean(1, keepdims=True)  # center each condition
    print(name, "attrs:", len(attrs), "gene rows:", n, flush=True)
    targets_all.extend(attrs)
    blocks.append(eff.astype(np.float16))

targets = np.array(targets_all)
effects = np.concatenate(blocks, axis=0)
cells = np.full(len(targets), 100.0, dtype=np.float32)
np.savez(f"{BASE}/pertseq/replogle_aux.npz",
         genes=np.array(panel), targets=targets,
         ctrl=np.zeros(G, dtype=np.float32),
         effects=effects, cells=cells)
print("saved replogle_aux.npz", effects.shape,
      "mean abs:", float(np.abs(effects.astype(np.float32)).mean()))
