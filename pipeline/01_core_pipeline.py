# -*- coding: utf-8 -*-
"""MAGWorld 核心单细胞管线（服务器端）
数据集: pbmc3k, aissa, dat17, xiehon, pancreas, dentategyrus
流程: 质控 -> 双细胞 -> 归一化 -> HVG -> PCA -> 聚类 -> UMAP -> 标志基因
输出: results/core/<name>_processed.h5ad, markers csv
"""
import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

DATA = "/nfs_beijing/zizhuo/vcc/data"
OUT = "/nfs_beijing/zizhuo/vcc/results/core"
os.makedirs(OUT, exist_ok=True)

MOUSE = {"pancreas", "dentategyrus"}


def load(name):
    return sc.read_h5ad(f"{DATA}/{name}.h5ad")


def qc_cluster(name, ad):
    ad.var_names_make_unique()
    mito_pat = "mt-" if name in MOUSE else "MT-"
    ad.var["mt"] = ad.var_names.str.startswith(mito_pat)
    sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], inplace=True, percent_top=None,
                               log1p=False)
    try:
        sc.pp.scrublet(ad, expected_doublet_rate=0.06, random_state=0)
        ad = ad[~ad.obs["predicted_doublet"]].copy()
    except Exception as e:
        print(f"[{name}] scrublet skipped: {e}")
    ad = ad[ad.obs["n_genes_by_counts"] > 200].copy()
    ad = ad[ad.obs["pct_counts_mt"] < (20 if name in MOUSE else 15)].copy()
    sc.pp.filter_genes(ad, min_cells=3)
    X = ad.X
    mx = X.max() if not hasattr(X, "toarray") else X.toarray().max()
    if mx < 20:  # 已是 log-normalized 数据
        print(f"[{name}] already normalized (max={mx:.2f}), skip normalize_total")
        pass
    else:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    ad.raw = ad
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat")
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(ad, resolution=0.8, key_added="leiden", random_state=0)
    sc.tl.umap(ad, random_state=0)
    sc.tl.rank_genes_groups(ad, "leiden", method="wilcoxon",
                            key_added="markers", pts=True)
    return ad


def main():
    for name in ["pbmc3k", "aissa", "dat17", "xiehon", "pancreas", "dentategyrus"]:
        try:
            print(f"===== {name} =====", flush=True)
            ad = load(name)
            print(f"raw: {ad.n_obs} cells x {ad.n_vars} genes; "
                  f"obs={list(ad.obs.columns)[:8]}", flush=True)
            ad = qc_cluster(name, ad)
            print(f"filtered: {ad.n_obs} cells x {ad.n_vars} genes", flush=True)
            ad.write_h5ad(f"{OUT}/{name}_processed.h5ad", compression="gzip")
            mk = sc.get.rank_genes_groups_df(ad, None, key="markers")
            mk.to_csv(f"{OUT}/{name}_markers.csv", index=False)
            print(f"[{name}] DONE", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{name}] FAILED: {e}", flush=True)
    print("ALL_CORE_DONE")


if __name__ == "__main__":
    main()
