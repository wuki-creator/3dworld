# -*- coding: utf-8 -*-
"""探查 xiehon / aissa / dat17 的 perturbation 标签格式"""
import scanpy as sc

for name in ["xiehon", "aissa", "dat17"]:
    try:
        ad = sc.read_h5ad(
            f"/nfs_beijing/zizhuo/vcc/results/core/{name}_processed.h5ad")
        p = ad.obs["perturbation"].astype(str)
        print(f"== {name}: {ad.n_obs} cells, {p.nunique()} perts", flush=True)
        print(sorted(p.unique())[:15], flush=True)
        print("counts head:", p.value_counts().head(8).to_dict(), flush=True)
    except Exception as e:
        print(name, "ERR", str(e)[:100], flush=True)
print("PROBE_DONE")
