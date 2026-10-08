# -*- coding: utf-8 -*-
"""Rebuild /nfs_beijing_os/zizhuo_vcc/extracted/ from h1_2025 originals.

The home-dir copy (/home/zizhuo/vcc_data/extracted) was wiped by a disk
cleanup.  Everything here is reconstructed on NFS so it cannot be lost again:
  - gene_names.csv   : official 18,533-gene panel (from signatures npz)
  - pert_counts.csv  : 150 training perts (original csv) + 50 validation perts
  - context_A/B/C.h5ad : non-targeting control cells from adata_Training.h5ad,
                         split into 3 contexts round-robin by batch
"""
import csv
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

H1 = "/nfs_beijing_os/zizhuo_vcc/h1_2025"
OUT = Path("/nfs_beijing_os/zizhuo_vcc/extracted")
OUT.mkdir(parents=True, exist_ok=True)

# ---- gene panel ----
sig = np.load("/nfs_beijing_os/zizhuo_vcc/signatures/h1_trainval_signatures.npz")
panel = [str(g) for g in sig["genes"]]
(OUT / "gene_names.csv").write_text("gene_name\n" + "\n".join(panel) + "\n")
print("gene_names.csv:", len(panel), flush=True)

# ---- pert counts: training csv + validation targets ----
train_pc = pd.read_csv(f"{H1}/pert_counts_Training.csv")
val = ad.read_h5ad(f"{H1}/adata_Validation.h5ad", backed="r")
val_counts = val.obs["target_gene"].value_counts()
val.obs  # touch
val_rows = [
    {"target_gene": g, "n_cells": int(val_counts.get(g, 0)),
     "median_umi_per_cell": float("nan")}
    for g in sorted(set(str(t) for t in val_counts.index) - {"non-targeting"})
]
val.file.close()
merged = pd.concat([train_pc, pd.DataFrame(val_rows)], ignore_index=True)
merged.to_csv(OUT / "pert_counts.csv", index=False)
print("pert_counts.csv:", len(merged), "perts", flush=True)

# ---- control cells -> context_A/B/C ----
a = ad.read_h5ad(f"{H1}/adata_Training.h5ad", backed="r")
obs = a.obs
ctrl_mask = (obs["target_gene"] == "non-targeting").values
ctrl_obs = obs[ctrl_mask].copy()
batches = sorted(ctrl_obs["batch"].astype(str).unique())
batch_ctx = {b: i % 3 for i, b in enumerate(batches)}
ctx_label = ctrl_obs["batch"].astype(str).map(batch_ctx).values
X = a[ctrl_mask].X  # SparseDataset -> load via to_memory on subset
X = X.to_memory() if hasattr(X, "to_memory") else X
var = a.var.copy()
a.file.close()
print("control cells:", X.shape, flush=True)

import scipy.sparse as sp
if not sp.issparse(X):
    X = sp.csr_matrix(X)
X = X.tocsr()

names = ["A", "B", "C"]
for ci, name in enumerate(names):
    m = ctx_label == ci
    sub = ad.AnnData(X=X[m], obs=ctrl_obs[m], var=var)
    sub.write_h5ad(OUT / f"context_{name}.h5ad", compression="gzip")
    print(f"context_{name}.h5ad:", m.sum(), "cells", flush=True)

print("rebuild done", flush=True)
