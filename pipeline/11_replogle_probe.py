# -*- coding: utf-8 -*-
"""探查 Replogle K562 essential h5ad 结构。"""
import numpy as np
import scanpy as sc

p = "/nfs_beijing/zizhuo/vcc/data/norman/ReplogleWeissman2022_K562_essential.h5ad"
ad = sc.read_h5ad(p)
print("shape:", ad.shape)
print("X:", type(ad.X), getattr(ad.X, "dtype", None))
xs = ad.X[:5]
if hasattr(xs, "toarray"):
    xs = xs.toarray()
print("X[:5] rowsum:", np.asarray(xs.sum(1)).ravel())
print("layers:", list(ad.layers.keys()))
print("obs columns:", ad.obs.columns.tolist())
for c in ad.obs.columns:
    try:
        vc = ad.obs[c].astype(str).value_counts()
        print(f"--- {c} ({len(vc)} uniq) ---")
        print(vc.head(6).to_string())
    except Exception as e:
        print(c, "err", e)
print("var columns:", ad.var.columns.tolist())
print("var head:", ad.var_names[:5].tolist())
if "perturbation" in ad.obs:
    per = ad.obs["perturbation"].astype(str)
    n_ctrl = (per == "control").sum()
    print("ctrl cells:", int(n_ctrl))
    s = per[(per != "control") & ~per.str.contains(r"\+|_")]
    print("single-gene cells:", int(len(s)), "uniq:", s.nunique())
print("PROBE_DONE")
