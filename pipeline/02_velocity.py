# -*- coding: utf-8 -*-
"""MAGWorld RNA 速率 / 轨迹分析（scVelo 动力学模型 + PAGA + 潜在时间）
从原始 counts 出发独立处理（scVelo 自行过滤/归一化），聚类标签继承核心管线。
适用数据集: pancreas（小鼠胰岛内分泌分化）, dentategyrus（海马神经发生）
"""
import os
import warnings
import numpy as np
import scanpy as sc
import scvelo as scv

warnings.filterwarnings("ignore")
scv.settings.verbosity = 1

DATA = "/nfs_beijing/zizhuo/vcc/data"
CORE = "/nfs_beijing/zizhuo/vcc/results/core"
OUT = "/nfs_beijing/zizhuo/vcc/results/velocity"
os.makedirs(OUT, exist_ok=True)


def run(name):
    ad = sc.read_h5ad(f"{DATA}/{name}.h5ad")
    core = sc.read_h5ad(f"{CORE}/{name}_processed.h5ad")
    ad.var_names_make_unique()
    scv.pp.filter_genes(ad, min_shared_counts=20)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat")
    ad = ad[:, ad.var["highly_variable"]].copy()
    scv.pp.moments(ad, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(ad, n_jobs=16)
    scv.tl.velocity(ad, mode="dynamical")
    scv.tl.velocity_graph(ad, n_jobs=16)
    scv.tl.latent_time(ad)
    # 继承核心管线聚类标签（按细胞交集）
    common = ad.obs_names.intersection(core.obs_names)
    ad.obs["leiden"] = "NA"
    ad.obs.loc[common, "leiden"] = core.obs.loc[common, "leiden"].astype(str)
    ad = ad[ad.obs["leiden"] != "NA"].copy()
    scv.tl.paga(ad, groups="leiden")
    ad.write_h5ad(f"{OUT}/{name}_velocity.h5ad", compression="gzip")
    scv.pl.velocity_embedding_stream(ad, basis="umap", color="leiden",
                                     save=f"{OUT}/{name}_velocity_stream.png",
                                     show=False, legend_fontsize=6,
                                     title=f"{name}: RNA velocity")
    scv.pl.latent_time(ad, color="leiden", save=f"{OUT}/{name}_latent_time.png",
                       show=False, title=f"{name}: latent time")
    scv.pl.paga(ad, save=f"{OUT}/{name}_paga.png", show=False)
    print(f"[{name}] velocity DONE", flush=True)


if __name__ == "__main__":
    for name in ["pancreas", "dentategyrus"]:
        try:
            run(name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{name}] FAILED: {e}", flush=True)
    print("ALL_VELOCITY_DONE")
