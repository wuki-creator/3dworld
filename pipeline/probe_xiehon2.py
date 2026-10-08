# -*- coding: utf-8 -*-
"""深度探查 xiehon：全部 obs 列 + 扰动名模式分类"""
import re
import scanpy as sc

ad = sc.read_h5ad("/nfs_beijing/zizhuo/vcc/data/xiehon.h5ad")
print("raw obs columns:", list(ad.obs.columns), flush=True)
for c in ad.obs.columns:
    v = ad.obs[c].astype(str)
    print(f"  {c}: {v.nunique()} uniq | {sorted(v.unique())[:6]}", flush=True)

p = ad.obs["perturbation"].astype(str)
pat_gene = re.compile(r"^[A-Za-z0-9_.-]+_\d+$")
gene_like = sorted([x for x in p.unique() if pat_gene.match(x)
                    and not x.lower().startswith(("gfp", "cag"))])
coord_like = [x for x in p.unique() if x.startswith("chr")]
print(f"\npert total uniq: {p.nunique()}", flush=True)
print(f"gene-like (Name_N): {len(gene_like)}", flush=True)
print(gene_like[:40], flush=True)
print(f"coordinate-like: {len(coord_like)}", flush=True)
vc = p.value_counts()
print("top20 counts:", vc.head(20).to_dict(), flush=True)
print("PROBE2_DONE")
