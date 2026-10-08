# -*- coding: utf-8 -*-
"""探查 scPerturb Norman h5ad 结构, 为 09 基准脚本定字段。"""
import sys
import numpy as np
import scanpy as sc

p = "/nfs_beijing/zizhuo/vcc/data/norman/NormanWeissman2019_filtered.h5ad"
ad = sc.read_h5ad(p)
print("shape:", ad.shape)
print("X type:", type(ad.X), "dtype:", getattr(ad.X, "dtype", None))
xs = ad.X[:5]
if hasattr(xs, "toarray"):
    xs = xs.toarray()
print("X[:5] sum per row:", np.asarray(xs.sum(1)).ravel())
print("layers:", list(ad.layers.keys()))
print("raw is None:", ad.raw is None)
print("obs columns:", ad.obs.columns.tolist())
for c in ad.obs.columns:
    try:
        vc = ad.obs[c].astype(str).value_counts()
        print(f"--- {c} ({len(vc)} uniq) ---")
        print(vc.head(8).to_string())
    except Exception as e:
        print(c, "err", e)
print("var columns:", ad.var.columns.tolist())
print("var_names head:", ad.var_names[:5].tolist())
print("PROBE_DONE")
