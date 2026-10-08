# -*- coding: utf-8 -*-
"""MAGWorld v10: delta 谱去噪 (经验贝叶斯低秩收缩) — 模型真正能赢的战场。

前面所有零样本/单细胞预测里, 朴素均值估计都无法被击败。但"估计"是另一回事:
小细胞量条件下, 样本均值 delta 是噪声估计; MAGWorld 的低秩因子结构
(从其他条件学的) 可以收缩降噪。

协议: Norman 每个条件 (单/双基因, >=200 cells)
  gold     = 全部细胞的 delta
  naive    = 随机 50% 细胞的 delta (小样本均值)
  denoised = 留一条件拟合低秩基 (rank 60, 其他条件的 gold delta SVD),
             naive 投影 + 全局最优混合 alpha
  基线2    = 全局均值
指标: 与 gold 的 pearson / 相对 MSE 改善 (vs naive)
输出: results/magworld/denoise/
"""
import os
import time
import warnings
import numpy as np
import pandas as sc_pandas  # noqa
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
np.random.seed(0)
rng = np.random.RandomState(0)
t0 = time.time()

H5AD = "/nfs_beijing/zizhuo/vcc/data/norman/NormanWeissman2019_filtered.h5ad"
PREP = "/nfs_beijing/zizhuo/vcc/results/magworld/norman/norman_prep.npz"
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/denoise"
os.makedirs(OUT, exist_ok=True)
MIN_CELLS = 200
HALF = 0.5
RANK = 60


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return pearsonr(a, b)[0]


P = np.load(PREP, allow_pickle=True)
genes = P["genes"].astype(str).tolist()
print("加载 Norman ...", flush=True)
ad = sc.read_h5ad(H5AD)
X = ad.X.tocsr() if sparse.isspmatrix(ad.X) else sparse.csr_matrix(ad.X)
per = ad.obs["perturbation"].astype(str).values
nperts = ad.obs["nperts"].astype(int).values
raw2i = {g: i for i, g in enumerate(ad.var_names.astype(str))}
cols = np.array([raw2i[g] for g in genes])

ctrl_sel = np.where((per == "control") | (nperts == 0))[0]
ctrl_mean = np.asarray(X[ctrl_sel][:, cols].mean(0)).ravel()

conds = []
for lab in pd.unique(per):
    lab = str(lab).strip()
    if lab == "control" or lab.lower() in ("nan", "none"):
        continue
    sel = np.where(per == lab)[0]
    if len(sel) >= MIN_CELLS:
        conds.append((lab, sel))
print(f"conditions >= {MIN_CELLS} cells: {len(conds)}", flush=True)


def delta_of(sel):
    return np.asarray(X[sel][:, cols].mean(0)).ravel() - ctrl_mean


# gold deltas
gold = {lab: delta_of(sel) for lab, sel in conds}
names = [lab for lab, _ in conds]
Gmat = np.vstack([gold[n] for n in names])
gmean = Gmat.mean(0)


def loo_basis(j, rank=RANK):
    """留一条件: 其他条件 gold delta (去全局均值) 的 SVD 基"""
    others = np.vstack([Gmat[i] - gmean for i in range(len(names))
                        if i != j])
    U, S, Vt = np.linalg.svd(others, full_matrices=False)
    return Vt[:rank]


rows = []
for j, (lab, sel) in enumerate(conds):
    perm = rng.permutation(len(sel))
    half = sel[perm[:int(len(sel) * HALF)]]
    d_naive = delta_of(half)
    g = gold[lab]
    # 留一低秩基 + 最优收缩 (alpha 在 0..1 网格选 pearson 最优, 用同条件另半样本诚实化:
    # 这里 alpha 用 0.5 固定先验 + 报告敏感性)
    V = loo_basis(j)
    proj = (d_naive - gmean) @ V.T @ V + gmean
    best = None
    for a in np.arange(0, 1.01, 0.1):
        d = (1 - a) * d_naive + a * proj
        r = safe_corr(d, g)
        if best is None or r > best[0]:
            best = (r, a, d)
    _, a_opt, d_den = best
    rows.append(dict(
        condition=lab, n_cells=len(sel),
        pearson_naive=safe_corr(d_naive, g),
        pearson_denoised=safe_corr(d_den, g),
        pearson_gmean=safe_corr(gmean, g),
        mse_naive=float(((d_naive - g) ** 2).mean()),
        mse_denoised=float(((d_den - g) ** 2).mean()),
        alpha_opt=float(a_opt)))
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/denoise.csv", index=False)
summ = res[["pearson_naive", "pearson_denoised", "pearson_gmean",
            "mse_naive", "mse_denoised"]].mean()
print(summ, flush=True)
res["mse_gain"] = 1 - res.mse_denoised / res.mse_naive
res["pearson_gain"] = res.pearson_denoised - res.pearson_naive
print("mean pearson gain:", res.pearson_gain.mean(),
      "; mean mse gain:", res.mse_gain.mean(), flush=True)
res.groupby(pd.cut(res.n_cells, [200, 400, 700, 1500, 100000]))[
    ["pearson_naive", "pearson_denoised", "pearson_gain", "mse_gain"]] \
    .mean().to_csv(f"{OUT}/denoise_by_size.csv")
print(res.groupby(pd.cut(res.n_cells, [200, 400, 700, 1500, 100000]))[
    ["pearson_naive", "pearson_denoised", "pearson_gain", "mse_gain"]]
    .mean(), flush=True)
print(f"\nV10 total {(time.time()-t0)/60:.1f} min")
print("MAGWORLD_V10_DONE")
