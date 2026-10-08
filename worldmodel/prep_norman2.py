# -*- coding: utf-8 -*-
"""Consume norman_spec.npz + downloaded matrix -> norman_aux.npz (panel-aligned)."""
import gzip
import numpy as np
from scipy.io import mmread
from scipy import sparse

BASE = "/nfs_beijing_os/zizhuo_vcc/pertseq/norman"
spec = np.load(f"{BASE}/norman_spec.npz", allow_pickle=True)
gpanel = spec["gpanel"]; ctrl_rows = spec["ctrl_rows"]
cond_names = spec["cond_names"]; cond_rows = [spec["cond_rows_flat"][spec["cond_ptr"][i]:spec["cond_ptr"][i+1]] for i in range(len(spec["cond_ptr"])-1)]

print("loading matrix...", flush=True)
mat = mmread(gzip.open(f"{BASE}/GSE133344_filtered_matrix.mtx.gz", "rt"))
mat = sparse.csr_matrix(mat)
print("matrix raw:", mat.shape)
if mat.shape[0] == len(gpanel):      # genes x cells (10x)
    X = mat.T.tocsr()
elif mat.shape[1] == len(gpanel):    # cells x genes
    X = mat
else:
    raise ValueError("matrix shape mismatch with gene table")
keep = gpanel >= 0
X = X[:, keep]
gmap = gpanel[keep]
print("cells x mapped genes:", X.shape)

tot = np.asarray(X.sum(1)).ravel()
scale = 1e4 / np.maximum(tot, 1.0)
X = sparse.diags(scale).dot(X)
X.data = np.log1p(X.data)

ctrl_mean = np.asarray(X[ctrl_rows].mean(0)).ravel()

M = len(cond_names)
G = len(gmap)
targets = cond_names
effects = np.zeros((M, G), dtype=np.float32)
cells = np.zeros(M, dtype=np.float32)
for i, sel in enumerate(cond_rows):
    effects[i] = np.asarray(X[sel].mean(0)).ravel() - ctrl_mean
    cells[i] = len(sel)

np.savez(f"{BASE}/norman_aux.npz",
         genes=gmap.astype(np.int32),
         targets=targets,
         ctrl=ctrl_mean.astype(np.float32),
         effects=effects,
         cells=cells)
print("saved norman_aux.npz", effects.shape, "delta mean abs:", float(np.abs(effects).mean()),
      "ctrl mean:", float(ctrl_mean.mean()))
