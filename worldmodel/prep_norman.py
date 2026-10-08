# -*- coding: utf-8 -*-
"""Prepare Norman 2019 Perturb-seq as auxiliary pretraining signatures.

Input (GEO GSE133344 filtered files):
  matrix.mtx.gz            cells x genes raw counts (10x mtx)
  genes/barcodes/cell_identities
Output: /nfs_beijing_os/zizhuo_vcc/pertseq/norman_aux.npz
  genes      (G,)   panel symbols (same order as H1 signatures)
  targets    (M,)   perturbed gene symbol per condition (single-gene KOs only)
  ctrl       (G,)   control mean profile (log1p CP10k)
  effects    (M,G)  KO mean - ctrl mean
  cells      (M,)   number of cells per condition
"""
import csv
import gzip
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse

BASE = "/nfs_beijing_os/zizhuo_vcc/pertseq/norman"
SIG = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_trainval_signatures.npz"

# panel genes from H1 signatures
panel = np.load(SIG)["genes"].astype(str).tolist()
pidx = {g: i for i, g in enumerate(panel)}

# HGNC symbol normalization
approved = set(); syn2sym = {}
with open("/nfs_beijing_os/zizhuo_vcc/embeddings/hgnc_complete_set.txt", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    i_sym = header.index("symbol"); i_prev = header.index("prev_symbol"); i_alias = header.index("alias_symbol")
    for row in rd:
        if len(row) <= max(i_sym, i_prev, i_alias):
            continue
        sym = row[i_sym].strip()
        if not sym:
            continue
        approved.add(sym)
        for o in (row[i_prev] + "|" + row[i_alias]).split("|"):
            o = o.strip()
            if o:
                syn2sym[o.upper()] = sym
def canon(g):
    g = g.strip().upper()
    return g if g in approved else syn2sym.get(g, g)

print("loading genes...", flush=True)
ng = pd.read_csv(f"{BASE}/GSE133344_filtered_genes.tsv.gz", sep="\t", header=None)
ng.columns = ["gene", "id"][: len(ng.columns)]
ng["panel_idx"] = ng["gene"].map(lambda g: pidx.get(canon(str(g)), -1))
keep = ng["panel_idx"].values >= 0
print(f"genes in file: {len(ng)}, mapped to panel: {keep.sum()}")

print("loading identities...", flush=True)
ident = pd.read_csv(f"{BASE}/GSE133344_filtered_cell_identities.csv.gz")
print(ident.columns.tolist(), ident.shape)
print(ident.head(3).to_string()[:500])

print("loading matrix...", flush=True)
mat = mmread(gzip.open(f"{BASE}/GSE133344_filtered_matrix.mtx.gz", "rt"))
mat = sparse.csr_matrix(mat)
print("matrix:", mat.shape, "nnz:", mat.nnz)
# mtx orientation: usually genes x cells in 10x; detect
if mat.shape[0] == len(ng):
    X = mat.T.tocsr()
else:
    X = mat
assert X.shape[1] == len(ng), (X.shape, len(ng))
X = X[:, keep]
gmap = ng["panel_idx"].values[keep]
print("cells:", X.shape[0], "panel genes kept:", X.shape[1])

# log1p CP10k per cell
tot = np.asarray(X.sum(1)).ravel()
scale = 1e4 / np.maximum(tot, 1.0)
X = sparse.diags(scale).dot(X)
X.data = np.log1p(X.data)

# align identities to barcodes
bc = pd.read_csv(f"{BASE}/GSE133344_filtered_barcodes.tsv.gz", header=None)[0].astype(str).str.replace("-[0-9]+$", "", regex=True)
bc_order = {b: i for i, b in enumerate(bc)}
key = ident.columns[0]
ident[key] = ident[key].astype(str).str.replace("-[0-9]+$", "", regex=True)
ident["row"] = ident[key].map(bc_order)
ident = ident.dropna(subset=["row"])
ident["row"] = ident["row"].astype(int)
print("cells with identity:", len(ident))

# find perturbation label column
cand = [c for c in ident.columns if c != "row" and c != key]
print("identity columns:", cand)

# Norman format: 'gene' column like 'CEBPA' or 'CEBPA+GATA2'; control: 'AAVS1'/'negctrl'/'NT'
gcol = None
for c in cand:
    vals = ident[c].astype(str).head(2000)
    if vals.str.contains(r"\+|AAVS1|neg|NT", case=False, regex=True).any():
        gcol = c
        break
if gcol is None:
    gcol = cand[0]
print("using label column:", gcol)
labels = ident[gcol].astype(str).str.strip()
is_ctrl = labels.str.contains(r"AAVS1|negctrl|non-targeting|^NT|^CTRL", case=False, regex=True)
print("ctrl cells:", int(is_ctrl.sum()), "label samples:", labels.value_counts().head(8).to_dict())

rows = ident["row"].values
ctrl_mean = np.asarray(X[rows[is_ctrl.values]].mean(0)).ravel() if is_ctrl.sum() else np.asarray(X.mean(0)).ravel()

singles = {}
lab_arr = labels.values
for lab in pd.unique(lab_arr):
    if "+" in lab or is_ctrl.loc[labels == lab].iloc[0] if False else None:
        pass
ctrl_set = set(labels[is_ctrl].unique())
for lab in pd.unique(lab_arr):
    if lab in ctrl_set or "+" in lab or not lab:
        continue
    cg = canon(lab)
    if cg not in pidx:
        continue
    sel = rows[lab_arr == lab]
    if len(sel) < 20:
        continue
    singles[cg] = (sel, len(sel))

print("usable single-KO conditions:", len(singles))
targets = sorted(singles)
effects = np.zeros((len(targets), len(panel)), dtype=np.float32)
cells = np.zeros(len(targets), dtype=np.float32)
for i, tg in enumerate(targets):
    sel, n = singles[tg]
    effects[i] = np.asarray(X[sel].mean(0)).ravel() - ctrl_mean
    cells[i] = n

out = {
    "genes": np.array(panel),
    "targets": np.array(targets),
    "ctrl": ctrl_mean.astype(np.float32),
    "effects": effects,
    "cells": cells,
}
np.savez("/nfs_beijing_os/zizhuo_vcc/pertseq/norman_aux.npz", **out)
print("saved norman_aux.npz", effects.shape, "delta mean abs:", float(np.abs(effects).mean()))
