# -*- coding: utf-8 -*-
"""探查 dat17 的 target 列 + 细胞数分布"""
import scanpy as sc

ad = sc.read_h5ad("/nfs_beijing/zizhuo/vcc/results/core/dat17_processed.h5ad")
print("obs:", list(ad.obs.columns), flush=True)
t = ad.obs["target"].astype(str)
p = ad.obs["perturbation"].astype(str)
print("target uniq:", t.nunique(), flush=True)
print(sorted(t.unique())[:20], flush=True)
import pandas as pd
df = pd.DataFrame({"p": p, "t": t}).drop_duplicates()
print("p->t mapping rows:", len(df), flush=True)
print(df.head(15).to_dict("records"), flush=True)
vc = p.value_counts()
print("perts>=60:", (vc >= 60).sum(), flush=True)
print("PROBE3_DONE")
