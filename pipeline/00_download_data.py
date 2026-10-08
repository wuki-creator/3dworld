# -*- coding: utf-8 -*-
"""MAGWorld 数据集下载脚本（在服务器上运行）。

数据集清单（5 个公开单细胞数据集）：
1. PBMC 3k (10x Genomics, Zheng et al. 2017)          — scanpy 内置
2. Kang et al. 2018 PBMC (IFN-beta 扰动, ~29k cells)   — figshare (scGen 处理版)
3. Pancreas endocrinogenesis (Bastidas-Ponce et al. 2019) — scvelo 内置
4. Dentate gyrus neurogenesis (Hochgerner et al. 2018)    — scvelo 内置
5. Paul et al. 2015 myeloid progenitors (mouse)           — scanpy 内置
"""
import os
import urllib.request
import scanpy as sc
import scvelo as scv

DATA = "/nfs_beijing/zizhuo/vcc/data"
os.makedirs(DATA, exist_ok=True)
sc.settings.datasetdir = DATA
scv.settings.datasetdir = DATA

KANG_URL = "https://ndownloader.figshare.com/files/16257757"


def dl_pbmc():
    ad = sc.datasets.pbmc3k()
    ad.var_names_make_unique()
    ad.write_h5ad(f"{DATA}/pbmc3k.h5ad")
    return "pbmc3k", ad.n_obs, ad.n_vars


def dl_kang():
    out = f"{DATA}/kang_raw.h5ad"
    if not os.path.exists(out):
        urllib.request.urlretrieve(KANG_URL, out)
    ad = sc.read_h5ad(out)
    ad.var_names_make_unique()
    ad.obs["condition"] = ad.obs["condition"].astype(str)
    ad.write_h5ad(f"{DATA}/kang.h5ad")
    return "kang", ad.n_obs, ad.n_vars


def dl_pancreas():
    ad = scv.datasets.pancreas()
    ad.var_names_make_unique()
    ad.write_h5ad(f"{DATA}/pancreas.h5ad")
    return "pancreas", ad.n_obs, ad.n_vars


def dl_dg():
    ad = scv.datasets.dentategyrus()
    ad.var_names_make_unique()
    ad.write_h5ad(f"{DATA}/dentategyrus.h5ad")
    return "dentategyrus", ad.n_obs, ad.n_vars


def dl_paul():
    ad = sc.datasets.paul15()
    ad.var_names_make_unique()
    ad.write_h5ad(f"{DATA}/paul15.h5ad")
    return "paul15", ad.n_obs, ad.n_vars


if __name__ == "__main__":
    for fn in (dl_pbmc, dl_kang, dl_pancreas, dl_dg, dl_paul):
        try:
            print("OK", fn(), flush=True)
        except Exception as e:
            print("FAIL", fn.__name__, repr(e), flush=True)
    print("ALL_DONE")
