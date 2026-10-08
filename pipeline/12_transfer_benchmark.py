# -*- coding: utf-8 -*-
"""MAGWorld v8: Replogle (2000 基因) -> Norman 跨数据集零样本迁移基准。

动机: v6/v7 证明 27~68 个训练基因无法学出基因特异结构 (判别排名=随机)。
Replogle 2022 K562 essential 在 K562 里扰动了 ~2000 个基因, 规模足以学习
"基因嵌入 -> 特异响应结构" 的映射, 然后零样本迁移到 Norman 2019 (CRISPRa)
的未见基因与未见双基因组合。

设计:
  特征 (数据集无关): CollecTRI PPR 平滑调控子 PCA (跨数据集同空间)
  核 (数据集无关): Norman 对照共表达相关矩阵 G (K562 同细胞系)
  算子:
    T1 Ridge-transfer:  E_regulon -> Replogle 全 delta 岭回归
    T2 Factor-transfer: Replogle 特异分量 (去均值) SVD 低秩 k,
                        E_regulon -> 因子得分岭回归, 预测 + Norman 均值
    T3 NW-corr-transfer: G_norman 相似度加权 Replogle 特异分量 + Norman 均值
    T4 Stack: 在 Norman 68 训练基因上 LOO 拟合 [均值, T2, T3, -prior] NNLS 组合
  基线: Norman-68 MeanDelta / Replogle 均值 / Identity / 置换零模型
  组合: T4 单基因预测加和 -> 131 个双基因测试; 协同统计
输出: results/magworld/transfer/
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

REP = "/nfs_beijing/zizhuo/vcc/data/norman/ReplogleWeissman2022_K562_essential.h5ad"
NORMAN_PREP = "/nfs_beijing/zizhuo/vcc/results/magworld/norman/norman_prep.npz"
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/transfer"
os.makedirs(OUT, exist_ok=True)
MIN_CELLS = 60
MAX_CELLS_PER_COND = 1500
FACT_KS = [30, 60, 100]


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return pearsonr(a, b)[0]


def eval_all(pred, true, meanv):
    scale = np.abs(true).mean() + 1e-9
    pc, tc = pred - meanv, true - meanv
    spec = safe_corr(pc, tc) if np.std(pc) > 1e-12 and np.std(tc) > 1e-12 \
        else np.nan
    return dict(delta_pearson=safe_corr(pred, true),
                specific_pearson=spec,
                delta_mae_norm=np.abs(pred - true).mean() / scale,
                deg_top50_overlap=len(
                    set(np.argsort(np.abs(true))[-50:]) &
                    set(np.argsort(np.abs(pred))[-50:])) / 50)


def ridge_fit(E_tr, D_tr, lams=(0.1, 1.0, 10.0, 100.0, 1000.0)):
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


def disc_ranks(pred_d, true_d):
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


# ================= 1. Norman 侧 (复用 09 资产) =================
print("加载 Norman prep ...", flush=True)
P = np.load(NORMAN_PREP, allow_pickle=True)
genes = P["genes"].astype(str).tolist()
g2i = {g: i for i, g in enumerate(genes)}
E_re = P["E_re"]          # 数据集无关特征 (regulon PCA)
E = P["E"]                # coexpr+regulon+intercept (Norman)
G = P["G"]                # Norman ctrl 基因-基因相关
m_tr = P["m_tr"]          # Norman 训练集均值
train_genes = P["train_genes"].astype(str).tolist()
test_genes = P["test_genes"].astype(str).tolist()
dkeys = P["delta_keys"].astype(str).tolist()
dmat = P["delta_mat"]
delta_n = {k: dmat[i] for i, k in enumerate(dkeys)}
print(f"norman: {len(genes)} genes, train={len(train_genes)} "
      f"test={len(test_genes)}", flush=True)

# ================= 2. Replogle 侧 =================
print("加载 Replogle ...", flush=True)
ad = sc.read_h5ad(REP)
X = ad.X.tocsr() if sparse.isspmatrix(ad.X) else np.asarray(ad.X)
per = ad.obs["perturbation"].astype(str).values
var_names = ad.var_names.astype(str).values
print(f"replogle: {X.shape} {'sparse' if sparse.isspmatrix(X) else 'dense'}",
      flush=True)

# 判断是否原始计数
xs = X[:20]
if hasattr(xs, "toarray"):
    xs = xs.toarray()
row_med = float(np.median(np.asarray(xs.sum(1)).ravel()))
print(f"median rowsum={row_med:.0f}", flush=True)
if row_med > 50000:  # 原始计数 -> log1p CP10k
    tot = np.asarray(X.sum(1)).ravel()
    scale = 1e4 / np.maximum(tot, 1.0)
    X = sparse.diags(scale).dot(X)
    X.data = np.log1p(X.data)
    print("已做 log1p CP10k 归一化", flush=True)

ctrl_sel = np.where(per == "control")[0]
print(f"replogle ctrl={len(ctrl_sel)}", flush=True)
cond_pairs = []
for lab in pd.unique(per):
    lab = str(lab).strip()
    if lab == "control" or not lab or lab.lower() in ("nan", "none") \
            or "+" in lab or "_" in lab:
        continue
    sel = np.where(per == lab)[0]
    if len(sel) < MIN_CELLS:
        continue
    if len(sel) > MAX_CELLS_PER_COND:
        sel = rng.choice(sel, MAX_CELLS_PER_COND, replace=False)
    cond_pairs.append((lab, sel))
# 只保留在 Norman 基因空间内的 (有嵌入行)
cond_pairs = [(g, s) for g, s in cond_pairs if g in g2i]
cond_names = [g for g, _ in cond_pairs]
cond_idx = [s for _, s in cond_pairs]
print(f"usable conditions (in norman space): {len(cond_names)}", flush=True)
print("examples:", cond_names[:8], flush=True)

# 映射到 Norman 基因空间
raw2i = {g: i for i, g in enumerate(var_names)}
shared = [g for g in genes if g in raw2i]
print(f"shared genes with norman space: {len(shared)}/{len(genes)}", flush=True)
cols_n = np.array([g2i[g] for g in shared])
cols_r = np.array([raw2i[g] for g in shared])

def take_rows(mat, rows, cols):
    sub = mat[rows][:, cols]
    return sub.toarray() if sparse.isspmatrix(sub) else np.asarray(sub)


ctrl_Xr = take_rows(X, ctrl_sel, cols_r)
ctrl_mean_r = ctrl_Xr.mean(0)
D_rep = np.zeros((len(cond_names), len(genes)))
for i, sel in enumerate(cond_idx):
    D_rep[i, cols_n] = take_rows(X, sel, cols_r).mean(0) - ctrl_mean_r
cond_g2i = {g: i for i, g in enumerate(cond_names)}
rep_mean = D_rep.mean(0)
D_rep_c = D_rep - rep_mean
print(f"replogle deltas: {D_rep.shape}, "
      f"mean|delta|={np.abs(D_rep).mean():.4f}", flush=True)

# ================= 3. 迁移算子 =================
# 特征: regulon PCA (数据集无关) + 截距
Erep = np.hstack([E_re, np.ones((len(genes), 1))])
rep_train_idx = np.array([g2i[g] for g in cond_names])

# T1: ridge 全 delta 迁移
W_t1, lam_t1, _ = ridge_fit(Erep[rep_train_idx], D_rep)
print(f"T1 ridge lam={lam_t1}", flush=True)

# T2: factor 特异分量迁移 (多种 k, 用 Norman train LOO 选)
Uf, Sf, Vtf = np.linalg.svd(D_rep_c, full_matrices=False)
print(f"replogle specific SVD: {Uf.shape}", flush=True)


def factor_model(k):
    Wk, lamk, _ = ridge_fit(Erep[rep_train_idx],
                             Uf[:, :k] * Sf[:k])
    def pred(g):
        scores = (Erep[[g2i[g]]] @ Wk)[0]
        return m_tr + scores @ Vtf[:k]
    return pred, lamk


factor_preds = {}
for k in FACT_KS:
    p_fn, lamk = factor_model(k)
    factor_preds[k] = p_fn
    print(f"T2 factor k={k} lam={lamk}", flush=True)

# T3: NW-corr 迁移 (Norman G 核, Replogle 特异分量)
def nw_transfer(g, tau):
    row = np.clip(G[g2i[g]], 0, None)
    w = np.exp(row[rep_train_idx] / tau)
    w = w / (w.sum() + 1e-9)
    return m_tr + w @ D_rep_c


# 用 Norman 训练基因 LOO 选 T2 k / T3 tau / 堆叠系数
taus = [0.02, 0.05, 0.1, 0.2, 0.4]
norman_train_in_g2i = [g for g in train_genes if g in g2i]
print(f"norman train genes usable: {len(norman_train_in_g2i)}", flush=True)

t3_scores = {}
for tau in taus:
    rs = [safe_corr(nw_transfer(g, tau), delta_n[g])
          for g in norman_train_in_g2i]
    t3_scores[tau] = np.nanmean(rs)
    print(f"T3 tau={tau}: norman-train r={t3_scores[tau]:.4f}", flush=True)
best_tau = max(t3_scores, key=t3_scores.get)

t2_scores = {}
for k in FACT_KS:
    rs = [safe_corr(factor_preds[k](g), delta_n[g])
          for g in norman_train_in_g2i]
    t2_scores[k] = np.nanmean(rs)
    print(f"T2 k={k}: norman-train r={t2_scores[k]:.4f}", flush=True)
best_k = max(t2_scores, key=t2_scores.get)

# T4 堆叠: [m_tr, T2, T3, -prior] NNLS (Norman train LOO)
net = pd.read_csv(
    "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
)[["source", "target", "weight"]].dropna()
tfs = sorted(net.source.unique())
tf2i = {t: i for i, t in enumerate(tfs)}
Mp = np.zeros((len(genes), len(tfs)))
for s, tg, w in net.itertuples(index=False):
    if tg in g2i and s in tf2i:
        Mp[g2i[tg], tf2i[s]] += w
prior_col = {g: Mp[:, tf2i[g]].copy() for g in genes if g in tf2i}

Arows, brows = [], []
for g in norman_train_in_g2i:
    feats = [m_tr, factor_preds[best_k](g), nw_transfer(g, best_tau)]
    pr = prior_col.get(g)
    feats.append(-pr if pr is not None else np.zeros(len(genes)))
    Arows.append(np.vstack(feats).T)
    brows.append(delta_n[g])
coef_stack, _ = nnls(np.vstack(Arows), np.concatenate(brows))
print(f"T4 stack coef = {coef_stack}", flush=True)


def stack_pred(g):
    pr = prior_col.get(g)
    return (coef_stack[0] * m_tr
            + coef_stack[1] * factor_preds[best_k](g)
            + coef_stack[2] * nw_transfer(g, best_tau)
            + coef_stack[3] * (-pr if pr is not None else 0))

# ================= 4. Norman 未见基因测试 =================
print("\n=== transfer: unseen single-gene test (Norman) ===", flush=True)
rows, pred_all, true_all = [], {}, {}
test_usable = [g for g in test_genes if g in g2i and g in delta_n]
for g in test_usable:
    true = delta_n[g]
    cands = {
        "MAGWorld-transfer-full": (Erep[[g2i[g]]] @ W_t1)[0],
        f"MAGWorld-factor-T2k{best_k}": factor_preds[best_k](g),
        f"MAGWorld-NW-T3t{best_tau}": nw_transfer(g, best_tau),
        "MAGWorld-stack": stack_pred(g),
        "MeanDelta-norman68": m_tr,
        "MeanDelta-replogle": rep_mean,
        "Identity": np.zeros_like(true),
    }
    pred_all[g] = cands
    true_all[g] = true
    for tag, p in cands.items():
        rows.append(dict(perturbation=g, model=tag,
                         **eval_all(p, true, m_tr)))
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/transfer_singles.csv", index=False)
summ = res.groupby("model")[["delta_pearson", "specific_pearson",
                             "delta_mae_norm",
                             "deg_top50_overlap"]].mean()
summ.to_csv(f"{OUT}/transfer_singles_summary.csv")
print(summ, flush=True)

disc_rows = []
for tag in res.model.unique():
    pr = {g: pred_all[g][tag] for g in test_usable}
    rk = disc_ranks(pr, true_all)
    for g, r in rk.items():
        disc_rows.append(dict(model=tag, perturbation=g, rank=r,
                              top1=int(r == 1), top5=int(r <= 5),
                              top10=int(r <= 10)))
disc = pd.DataFrame(disc_rows)
disc.to_csv(f"{OUT}/transfer_discrimination.csv", index=False)
dsumm = disc.groupby("model")[["rank", "top1", "top5", "top10"]].mean()
dsumm.to_csv(f"{OUT}/transfer_discrimination_summary.csv")
print(dsumm, flush=True)

# 置换零模型 (stack)
Wf_null, _, _ = ridge_fit(Erep[rep_train_idx], Uf[:, :best_k] * Sf[:best_k])
nulls = []
for g in test_usable:
    cs = []
    for _ in range(50):
        ri = rng.randint(0, len(Erep))
        while ri == g2i[g]:
            ri = rng.randint(0, len(Erep))
        e_row = Erep[[ri]].copy()
        e_row[0, -1] = 1.0
        pr = prior_col.get(g)
        spec_i = (e_row @ Wf_null)[0] @ Vtf[:best_k]
        p = (coef_stack[0] * m_tr
             + coef_stack[1] * (m_tr + spec_i)
             + coef_stack[2] * nw_transfer(g, best_tau)
             + coef_stack[3] * (-pr if pr is not None else 0))
        cs.append(safe_corr(p, delta_n[g]))
    nulls.append(dict(perturbation=g, delta_pearson=np.nanmean(cs)))
pd.DataFrame(nulls).to_csv(f"{OUT}/transfer_permuted_null.csv", index=False)
print("permuted null saved", flush=True)

# ================= 5. 双基因组合泛化 (预测单基因加和) =================
print("\n=== transfer: double-gene compositional test ===", flush=True)
def pred_single(g):
    if g in test_usable:
        return pred_all[g]["MAGWorld-stack"]
    if g in g2i:
        return stack_pred(g)
    return None

crows, syn_rows = [], []
m2 = 2 * m_tr
for lab in dkeys:
    if "+" not in lab:
        continue
    a_, b_ = lab.split("+")
    if lab not in delta_n or a_ not in g2i or b_ not in g2i:
        continue
    true = delta_n[lab]
    da, db = pred_single(a_), pred_single(b_)
    if da is None or db is None:
        continue
    cands = {"MAGWorld-stack-add": da + db,
             "MeanDelta-add": 2 * m_tr,
             "MeanDeltaRep-add": 2 * rep_mean}
    d_oa, d_ob = delta_n.get(a_), delta_n.get(b_)
    if d_oa is not None and d_ob is not None:
        cands["ObsAdditivity"] = d_oa + d_ob
    for tag, p in cands.items():
        crows.append(dict(combo=lab, model=tag, **eval_all(p, true, m2)))
    if d_oa is not None and d_ob is not None:
        def syn(dc, d1, d2):
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            return abs(np.linalg.norm(dc) - n1 - n2) / (n1 + n2 + 1e-9)
        syn_rows.append(dict(combo=lab,
                             synergy_obs=float(syn(true, d_oa, d_ob)),
                             synergy_model=float(syn(da + db, da, db))))
cres = pd.DataFrame(crows)
cres.to_csv(f"{OUT}/transfer_doubles.csv", index=False)
csumm = cres.groupby("model")[["delta_pearson", "specific_pearson",
                               "delta_mae_norm",
                               "deg_top50_overlap"]].mean()
csumm.to_csv(f"{OUT}/transfer_doubles_summary.csv")
print(csumm, flush=True)
syn_df = pd.DataFrame(syn_rows)
syn_df.to_csv(f"{OUT}/transfer_synergy.csv", index=False)
if len(syn_df):
    print(f"synergy: mean|err|={np.abs(syn_df.synergy_model - syn_df.synergy_obs).mean():.4f}"
          f"; obs={syn_df.synergy_obs.mean():.4f} model={syn_df.synergy_model.mean():.4f}",
          flush=True)

json.dump({"best_k": int(best_k), "best_tau": float(best_tau),
           "lam_t1": float(lam_t1),
           "n_rep_conditions": len(cond_names),
           "stack_coef": [float(c) for c in coef_stack]},
          open(f"{OUT}/transfer_config.json", "w"), indent=1)
print(f"\nV8 total {(time.time()-t0)/60:.1f} min")
print("MAGWORLD_V8_DONE")
