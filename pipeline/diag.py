# -*- coding: utf-8 -*-
"""诊断 aissa/dat17 的 obs 结构与扰动分布"""
import scanpy as sc
import pandas as pd

CORE = "/nfs_beijing/zizhuo/vcc/results/core"
for name in ["aissa", "dat17"]:
    ad = sc.read_h5ad(f"{CORE}/{name}_processed.h5ad")
    print("=====", name, ad.n_obs, "x", ad.n_vars, flush=True)
    print("obs columns:", list(ad.obs.columns), flush=True)
    print("var head:", list(ad.var_names[:5]), flush=True)
    for c in ["perturbation", "perturbation_name", "condition", "target_gene"]:
        if c in ad.obs.columns:
            vc = ad.obs[c].astype(str).value_counts()
            print(f"-- {c}: n_unique={ad.obs[c].nunique()}", flush=True)
            print(vc.head(8).to_string(), flush=True)
