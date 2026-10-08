# -*- coding: utf-8 -*-
"""MAGWorld v7: Norman 2019 (scPerturb harmonized) large-scale benchmark.

数据源: zenodo 7041849 / NormanWeissman2019_filtered.h5ad
        111,445 cells x 33,694 genes, X 已 log 归一化, K562 CRISPRa。
        obs['perturbation']: 'control' + 单基因符号 + 双基因 'A_B' (nperts 区分)。

设计 (见 v6 教训, 27 基因小基准信号到顶, Norman 提供 237 条件):
  - 对照: perturbation == 'control'  (11,855 cells)
  - 单基因 (nperts==1): seed0 划分 2/3 训练 / 1/3 未见测试
  - 双基因 (nperts==2): 全部用于组合泛化测试 (训练从不使用双基因)
  - 算子: ridge-v5 / NW-emb / NW-corr(3种) / RegulonPrior 机制混合 /
          收缩集成 (alpha 由训练 LOO 选)
  - 组合: 预测单基因之和 (纯零样本组合) vs 观测加性 (参考上界) vs 2x均值
  - 指标: delta_pearson / delta_mae_norm / DEG top50 overlap / 协同误差
输出: results/magworld/norman/
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
from scipy.optimize import nnls

warnings.filterwarnings("ignore")
np.random.seed(0)
rng = np.random.RandomState(0)
t0 = time.time()

H5AD = "/nfs_beijing/zizhuo/vcc/data/norman/NormanWeissman2019_filtered.h5ad"
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/norman"
os.makedirs(OUT, exist_ok=True)
EMB_DIM = 16
MIN_CELLS = 20
N_HVG = 3000


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return pearsonr(a, b)[0]


def eval_delta(pred, true):
    scale = np.abs(true).mean() + 1e-9
    return dict(delta_pearson=safe_corr(pred, true),
                delta_mae_norm=np.abs(pred - true).mean() / scale,
                deg_top50_overlap=len(
                    set(np.argsort(np.abs(true))[-50:]) &
                    set(np.argsort(np.abs(pred))[-50:])) / 50)


def spec_corr(pred, true, meanv):
    """特异分量相关: 剔除共享均值后的相关 (基因特异结构的金标准)"""
    pc = pred - meanv
    tc = true - meanv
    if np.std(pc) < 1e-12 or np.std(tc) < 1e-12:
        return np.nan
    return safe_corr(pc, tc)


def disc_ranks(pred_d, true_d):
    """判别排名: 预测-真实余弦相似矩阵中正确配对的升序名次 (1=最好)"""
    names = list(pred_d)
    P = np.vstack([pred_d[n] for n in names])
    T = np.vstack([true_d[n] for n in names])
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    S = Pn @ Tn.T
    ranks = []
    for i in range(len(names)):
        order = np.argsort(S[i])[::-1]
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    return dict(zip(names, ranks))


# ================= 1. 加载 (h5ad) =================
print("加载 Norman h5ad ...", flush=True)
ad = sc.read_h5ad(H5AD)
X = ad.X
if sparse.isspmatrix_csc(X):
    X = X.tocsr()
labels = ad.obs["perturbation"].astype(str).values
nperts = ad.obs["nperts"].astype(int).values
gene_names = ad.var_names.astype(str).values
print(f"cells={X.shape[0]}, genes={X.shape[1]}", flush=True)

is_ctrl = (labels == "control") | (nperts == 0)
ctrl_idx = np.where(is_ctrl)[0]
cond_idx = np.where(~is_ctrl)[0]
print(f"ctrl={len(ctrl_idx)}, pert={len(cond_idx)}", flush=True)

uniq = pd.unique(labels[cond_idx])
singles, doubles = {}, {}
for lab in uniq:
    lab = str(lab).strip()
    sel = cond_idx[labels[cond_idx] == lab]
    if len(sel) < MIN_CELLS:
        continue
    npt = nperts[sel[0]]
    if npt == 1:
        singles[lab] = sel
    elif npt == 2 and "_" in lab:
        parts = tuple(sorted(lab.split("_")))
        if len(parts) == 2 and all(p != lab for p in parts):
            doubles[parts] = sel
print(f"singles={len(singles)}, doubles={len(doubles)}", flush=True)
print("single examples:", sorted(singles)[:10], flush=True)
print("double examples:", sorted(doubles)[:5], flush=True)

# ================= 2. 基因空间 =================
g2raw = {g: i for i, g in enumerate(gene_names)}
# 过滤: 只保留矩阵里找得到的靶基因 (3 个符号已更名缺失)
singles = {g: s for g, s in singles.items() if g in g2raw}
doubles = {d: s for d, s in doubles.items()
           if all(g in g2raw for g in d)}
want = set(singles) | {g for d in doubles for g in d}
print(f"singles={len(singles)}, doubles={len(doubles)} (after symbol filter)",
      flush=True)
present = sorted(want)

sub = X[rng.choice(X.shape[0], min(25000, X.shape[0]), replace=False)]
var = np.asarray(sub.multiply(sub).mean(0)).ravel() - \
    np.asarray(sub.mean(0)).ravel() ** 2
hvg = [gene_names[i] for i in np.argsort(var)[::-1][:N_HVG]]
genes = sorted(set(hvg) | set(present))
g2i = {g: i for i, g in enumerate(genes)}
cols = np.array([g2raw[g] for g in genes])
print(f"gene space: {len(genes)}", flush=True)

ctrl_X = np.asarray(X[ctrl_idx][:, cols].todense())
ctrl_mean = ctrl_X.mean(0)
delta = {}
for lab, sel in singles.items():
    delta[lab] = np.asarray(X[sel][:, cols].todense()).mean(0) - ctrl_mean
for (a, b), sel in doubles.items():
    delta[f"{a}+{b}"] = np.asarray(X[sel][:, cols].todense()).mean(0) \
        - ctrl_mean

# ================= 3. 嵌入 =================
Xc = ctrl_X - ctrl_X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
E_co = Vt[:EMB_DIM].T * S[:EMB_DIM]
E_co = E_co / (np.linalg.norm(E_co, axis=1, keepdims=True) + 1e-9)

net = pd.read_csv(
    "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
)[["source", "target", "weight"]].dropna()
tfs = sorted(net.source.unique())
tf2i = {t: i for i, t in enumerate(tfs)}
M = np.zeros((len(genes), len(tfs)))
for s, tg, w in net.itertuples(index=False):
    if tg in g2i and s in tf2i:
        M[g2i[tg], tf2i[s]] += w
coln = np.abs(M).sum(0); coln[coln == 0] = 1.0
Psm = M / coln
Gsm = Psm @ Psm.T
Ms = M.copy()
for _ in range(20):
    Ms = 0.85 * (Gsm @ Ms) + 0.15 * M
U, S, Vt = np.linalg.svd(Ms - Ms.mean(0), full_matrices=False)
E_re = U[:, :8] * S[:8]
E_re = E_re / (np.linalg.norm(E_re, axis=1, keepdims=True) + 1e-9)
E_feat = np.hstack([E_co, E_re])
E_feat = E_feat / (np.linalg.norm(E_feat, axis=1, keepdims=True) + 1e-9)
E = np.hstack([E_feat, np.ones((len(genes), 1))])

sd = Xc.std(0); sd[sd < 1e-9] = 1.0
G = (Xc / sd).T @ (Xc / sd) / ctrl_X.shape[0]

prior_col = {g: M[:, tf2i[g]].copy() for g in present if g in tf2i}

# ================= 4. 算子 =================
def ridge_fit(E_tr, D_tr, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)):
    n = E_tr.shape[0]
    I = np.eye(E_tr.shape[1])
    best = (None, None, np.inf)
    for lam in lams:
        A = E_tr @ E_tr.T + lam * np.eye(n)
        Ainv = np.linalg.inv(A)
        H = E_tr @ E_tr.T @ Ainv
        denom = (1 - np.diag(H))[:, None]
        denom[np.abs(denom) < 1e-6] = 1e-6
        loo_pred = (H @ D_tr - np.diag(H)[:, None] * D_tr) / denom
        err = ((loo_pred - D_tr) ** 2).mean()
        if err < best[2]:
            W = np.linalg.solve(E_tr.T @ E_tr + lam * I, E_tr.T @ D_tr)
            best = (W, lam, err)
    return best


def nw_predict(row, tr_names, Dmat, tau):
    idx = [g2i[g] for g in tr_names]
    w = np.exp(row[idx] / max(tau, 1e-6))
    w = w / (w.sum() + 1e-9)
    return w @ Dmat


sg = sorted(singles)
perm = rng.permutation(len(sg))
n_te = max(8, len(sg) // 3)
test_genes = sorted(sg[i] for i in perm[:n_te])
train_genes = sorted(sg[i] for i in perm[n_te:])
print(f"split: train={len(train_genes)} test={len(test_genes)}", flush=True)
print("test genes:", test_genes, flush=True)
D_tr = np.vstack([delta[g] for g in train_genes])
m_tr = D_tr.mean(0)

# MeanDelta 训练 LOO 参考
loo_mean_rs = [safe_corr((len(train_genes) * m_tr - delta[g])
                         / (len(train_genes) - 1), delta[g])
               for g in train_genes]
print(f"MeanDelta train-LOO r={np.nanmean(loo_mean_rs):.4f}", flush=True)

# 训练集 LOO 选 NW 超参
taus = [0.02, 0.05, 0.1, 0.2, 0.4]
best = {}
for sim_name, rowfun in [
    ("emb", lambda g: E[:, :-1] @ E[g2i[g], :-1] /
     (np.linalg.norm(E[:, :-1], axis=1) *
      np.linalg.norm(E[g2i[g], :-1]) + 1e-9)),
    ("corr", lambda g: G[g2i[g]]),
    ("abscorr", lambda g: np.abs(G[g2i[g]])),
    ("poscorr", lambda g: np.clip(G[g2i[g]], 0, None)),
]:
    best_r, best_tau = -np.inf, taus[0]
    for tau in taus:
        rs = []
        for g in train_genes:
            tr = [x for x in train_genes if x != g]
            p = nw_predict(rowfun(g), tr,
                           np.vstack([delta[x] for x in tr]), tau)
            rs.append(safe_corr(p, delta[g]))
        r = np.nanmean(rs)
        if r > best_r:
            best_r, best_tau = r, tau
    best[f"NW-{sim_name}"] = (rowfun, best_tau, best_r)
    print(f"NW-{sim_name}: tau={best_tau} trainLOO r={best_r:.4f}", flush=True)

W_ridge, lam_r, _ = ridge_fit(E[[g2i[g] for g in train_genes]], D_tr)
print(f"ridge lam={lam_r}", flush=True)

# RegulonPrior 混合: pred = a*mean + b*(-P(g)) + c*ridge
Arows, brows = [], []
for g in train_genes:
    tr = [x for x in train_genes if x != g]
    Wl, _, _ = ridge_fit(E[[g2i[x] for x in tr]],
                         np.vstack([delta[x] for x in tr]))
    pr = (E[[g2i[g]]] @ Wl)[0]
    prior = prior_col.get(g)
    Arows.append(np.vstack([m_tr,
                            -prior if prior is not None else np.zeros(len(genes)),
                            pr]).T)
    brows.append(delta[g])
coef_h, _ = nnls(np.vstack(Arows), np.concatenate(brows))
print(f"hybrid coef (mean,-prior,ridge) = {coef_h}", flush=True)

# 收缩集成 alpha (best NW + MeanDelta)
nw_tag = max(best, key=lambda k: best[k][2])
rowfun, tau_nw, r_nw = best[nw_tag]
alphas = np.arange(0, 1.01, 0.1)
a_rs = []
for a in alphas:
    rs = []
    for g in train_genes:
        tr = [x for x in train_genes if x != g]
        p = a * nw_predict(rowfun(g), tr,
                           np.vstack([delta[x] for x in tr]), tau_nw) \
            + (1 - a) * m_tr
        rs.append(safe_corr(p, delta[g]))
    a_rs.append(np.nanmean(rs))
best_alpha = float(alphas[int(np.nanargmax(a_rs))])
print(f"blend: {nw_tag} alpha={best_alpha} trainLOO r={max(a_rs):.4f}",
      flush=True)

# Delta-factor 算子: 训练特异分量低秩 SVD, 嵌入->因子得分岭回归
D_c = D_tr - m_tr
Uf, Sf, Vtf = np.linalg.svd(D_c, full_matrices=False)
FACT_K = min(20, Uf.shape[1])
E_tr_idx = [g2i[g] for g in train_genes]
W_fact, lam_f, _ = ridge_fit(E[E_tr_idx], Uf[:, :FACT_K] * Sf[:FACT_K])
print(f"delta-factor: k={FACT_K} lam={lam_f}", flush=True)


def factor_pred(g):
    scores = (E[[g2i[g]]] @ W_fact)[0]
    return m_tr + scores @ Vtf[:FACT_K]


# ================= 5. 未见基因测试 =================
print("\n=== unseen single-gene test ===", flush=True)
rows, pred_singles, pred_all, true_all = [], {}, {}, {}
for g in test_genes:
    true = delta[g]
    pr_ridge = (E[[g2i[g]]] @ W_ridge)[0]
    prior = prior_col.get(g)
    pr_fact = factor_pred(g)
    cands = {
        "MAGWorld-ridge": pr_ridge,
        "MAGWorld-hybrid": coef_h[0] * m_tr
        + (coef_h[1] * (-prior) if prior is not None else 0)
        + coef_h[2] * pr_ridge,
        "MAGWorld-factor": pr_fact,
        nw_tag: nw_predict(rowfun(g), train_genes, D_tr, tau_nw),
        f"Blend-a{best_alpha}": best_alpha * nw_predict(
            rowfun(g), train_genes, D_tr, tau_nw) + (1 - best_alpha) * m_tr,
        "MeanDelta": m_tr,
        "Identity": np.zeros_like(true),
    }
    pred_all[g] = {t: p for t, p in cands.items()}
    true_all[g] = true
    for tag, p in cands.items():
        r = dict(perturbation=g, model=tag, **eval_delta(p, true))
        r["specific_pearson"] = spec_corr(p, true, m_tr)
        rows.append(r)
    pred_singles[g] = cands["MAGWorld-factor"]
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/norman_singles.csv", index=False)
summ = res.groupby("model")[["delta_pearson", "specific_pearson",
                             "delta_mae_norm",
                             "deg_top50_overlap"]].mean()
summ.to_csv(f"{OUT}/norman_singles_summary.csv")
print(summ, flush=True)

# 判别排名 (所有模型)
disc_rows = []
for tag in res.model.unique():
    pr = {g: pred_all[g][tag] for g in test_genes}
    rk = disc_ranks(pr, true_all)
    for g, r in rk.items():
        disc_rows.append(dict(model=tag, perturbation=g, rank=r,
                              top1=int(r == 1), top5=int(r <= 5)))
disc = pd.DataFrame(disc_rows)
disc.to_csv(f"{OUT}/norman_discrimination.csv", index=False)
dsumm = disc.groupby("model")[["rank", "top1", "top5"]].mean()
dsumm.to_csv(f"{OUT}/norman_discrimination_summary.csv")
print(dsumm, flush=True)

nulls = []
for g in test_genes:
    cs = []
    for _ in range(50):
        ri = rng.randint(0, len(E))
        while ri == g2i[g]:
            ri = rng.randint(0, len(E))
        e_row = E[[ri]].copy(); e_row[0, -1] = 1.0
        cs.append(safe_corr((e_row @ W_ridge)[0], delta[g]))
    nulls.append(dict(perturbation=g, delta_pearson=np.nanmean(cs)))
pd.DataFrame(nulls).to_csv(f"{OUT}/norman_permuted_null.csv", index=False)

# ================= 6. 双基因组合泛化 =================
print("\n=== double-gene compositional test ===", flush=True)
def pred_single(g):
    if g in pred_singles:
        return pred_singles[g]
    if g not in g2i:
        return None
    return best_alpha * nw_predict(rowfun(g), train_genes, D_tr, tau_nw) \
        + (1 - best_alpha) * m_tr

crows, syn_rows = [], []
m2 = 2 * m_tr
for (a_, b_), sel in sorted(doubles.items()):
    lab = f"{a_}+{b_}"
    true = delta[lab]
    da, db = pred_single(a_), pred_single(b_)
    cands = {"MAGWorld-add": da + db, "MeanDelta-add": 2 * m_tr}
    d_oa, d_ob = delta.get(a_), delta.get(b_)
    if d_oa is not None and d_ob is not None:
        cands["ObsAdditivity"] = d_oa + d_ob
    for tag, p in cands.items():
        r = dict(combo=lab, model=tag, **eval_delta(p, true))
        r["specific_pearson"] = spec_corr(p, true, m2)
        crows.append(r)
    if d_oa is not None and d_ob is not None:
        def syn(dc, d1, d2):
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            return abs(np.linalg.norm(dc) - n1 - n2) / (n1 + n2 + 1e-9)
        syn_rows.append(dict(combo=lab,
                             synergy_obs=float(syn(true, d_oa, d_ob)),
                             synergy_model=float(syn(da + db, da, db))))
cres = pd.DataFrame(crows)
cres.to_csv(f"{OUT}/norman_doubles.csv", index=False)
csumm = cres.groupby("model")[["delta_pearson", "specific_pearson",
                               "delta_mae_norm",
                               "deg_top50_overlap"]].mean()
csumm.to_csv(f"{OUT}/norman_doubles_summary.csv")
print(csumm, flush=True)
syn_df = pd.DataFrame(syn_rows)
syn_df.to_csv(f"{OUT}/norman_synergy.csv", index=False)
if len(syn_df):
    mae_syn = np.abs(syn_df.synergy_model - syn_df.synergy_obs).mean()
    print(f"synergy: mean|err|={mae_syn:.4f}; obs={syn_df.synergy_obs.mean():.4f}"
          f" model={syn_df.synergy_model.mean():.4f}", flush=True)

json.dump({"best_nw": nw_tag, "tau": float(tau_nw), "alpha": best_alpha,
           "lam_ridge": float(lam_r), "n_train": len(train_genes),
           "n_test": len(test_genes),
           "hybrid_coef": [float(c) for c in coef_h]},
          open(f"{OUT}/norman_config.json", "w"), indent=1)

# 保存供 12 跨数据集迁移脚本复用
np.savez(f"{OUT}/norman_prep.npz",
         genes=np.array(genes),
         E_co=E_co, E_re=E_re, E=E, G=G,
         m_tr=m_tr,
         D_tr=D_tr,
         train_genes=np.array(train_genes),
         test_genes=np.array(test_genes),
         delta_keys=np.array(sorted(delta)),
         delta_mat=np.vstack([delta[k] for k in sorted(delta)]))
print(f"\nV7 total {(time.time()-t0)/60:.1f} min")
print("MAGWORLD_V7_DONE")
