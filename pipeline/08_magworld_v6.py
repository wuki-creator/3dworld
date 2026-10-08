# -*- coding: utf-8 -*-
"""MAGWorld v6: operator zoo for gene-level zero-shot perturbation.

目标: 在 Datlinger2017 CRISPRi 基因级留一基准上超越 v5 (ridge, 0.243)
与 MeanDelta 基线 (0.257), 接近/达到同类方法 SOTA 水平。

改进:
  A. 算子动物园 (同一 seed-0 划分, 与 v5 完全可比):
     O1 线性岭回归 (v5 复现)
     O2 RBF 核岭回归 (嵌入空间)
     O3 Nadaraya-Watson (嵌入余弦相似度, tau 由训练集 LOO 选)
     O4 NW (对照共表达谱相似度核, sim in {corr, |corr|, max(corr,0)})
     O5 kNN 余弦插值 (k 由 LOO 选)
     O6 收缩集成: alpha*operator + (1-alpha)*MeanDelta, alpha 由训练集 LOO 选
     B1-B6: 上述各算子 + aissa 单药伪扰动增强 (EGFR/ALK/ROS1/MET)
  B. CollecTRI 调控子谱图扩散平滑 (PPR) 增强嵌入
  C. 5 折 CV (27 可用基因) 稳定选模型; 再在固定 9 基因测试集上评估
  D. MIN_CELLS=40 敏感性分析 (更多训练基因)
全部输出到 results/magworld/v6/, 不覆盖 v5 结果。
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

CORE = "/nfs_beijing/zizhuo/vcc/results/core"
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/v6"
os.makedirs(OUT, exist_ok=True)
EMB_DIM = 16
MIN_CELLS = 60
rng = np.random.RandomState(0)
t0 = time.time()


class VAE(nn.Module):
    def __init__(self, n_in, lat=32, hid=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU())
        self.mu = nn.Linear(hid, lat)
        self.lv = nn.Linear(hid, lat)
        self.dec = nn.Sequential(nn.Linear(lat, hid), nn.ReLU(),
                                 nn.Linear(hid, n_in))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.lv(h).clamp(-8, 4)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv

    def encode(self, x):
        return self.mu(self.enc(x))


def train_vae(Xnp, epochs, tag):
    X = torch.tensor(Xnp, dtype=torch.float32)
    model = VAE(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 1024):
            xb = X[perm[i:i + 1024]]
            xr, mu, lv = model(xb)
            rec = nn.functional.mse_loss(xr, xb)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = rec + 1e-3 * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += rec.item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"  [{tag}] {ep+1}/{epochs} recon={tot/n:.4f}", flush=True)
    return model


def safe_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return pearsonr(a, b)[0]


def coexpr_embedding(Xctrl, dim=EMB_DIM):
    Xc = Xctrl - Xctrl.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    E = Vt[:dim].T * S[:dim]
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def regulon_profile(genes, g2i):
    net = pd.read_csv(
        "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
    )[["source", "target", "weight"]].dropna()
    tfs = sorted(net.source.unique())
    tf2i = {t: i for i, t in enumerate(tfs)}
    M = np.zeros((len(genes), len(tfs)))
    for s, tg, w in net.itertuples(index=False):
        if tg in g2i and s in tf2i:
            M[g2i[tg], tf2i[s]] += w
    return M


def ppr_smooth(M, alpha=0.85, iters=20):
    """对调控子谱做列归一化图上的 PPR 平滑 (行=基因, 列=TF)"""
    S = M.copy()
    coln = np.abs(S).sum(0)
    coln[coln == 0] = 1.0
    P = S / coln  # 列随机
    # R <- alpha * (P P^T) R + (1-alpha) M, 迭代收敛 (PPR 平滑)
    R = M.copy()
    G = P @ P.T  # 基因-基因传播算子 (对称半正定)
    for _ in range(iters):
        R = alpha * (G @ R) + (1 - alpha) * M
    return R


def regulon_embedding_smoothed(genes, g2i, dim=8):
    M = regulon_profile(genes, g2i)
    if M.std() == 0:
        return np.zeros((len(genes), dim))
    Ms = ppr_smooth(M)
    U, S, Vt = np.linalg.svd(Ms - Ms.mean(0), full_matrices=False)
    k = min(dim, U.shape[1])
    R = U[:, :k] * S[:k]
    return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)


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


# ================= 数据加载 (与 v5 完全一致) =================
print("加载 dat17 ...", flush=True)
ad = sc.read_h5ad(f"{CORE}/dat17_processed.h5ad")
X = ad.raw.X.toarray() if hasattr(ad.raw.X, "toarray") else np.asarray(ad.raw.X)
X = pd.DataFrame(X, index=ad.obs_names, columns=ad.raw.var_names)
hv = ad.var["highly_variable"].reindex(X.columns).fillna(False).values
genes = list(X.columns[hv])

tgt = ad.obs["target"].astype(str)
tgt = tgt.where(tgt.notna() & (tgt.str.lower() != "nan"), "")
pert = ad.obs["perturbation"].astype(str)
is_ctrl = (tgt == "") | pert.str.lower().isin(["control", "ctrl", "nt",
                                                "non-targeting"])
all_targets = sorted(set(t for t in tgt if t))
extra = [t for t in all_targets if t in X.columns and t not in genes]
genes += extra
# v6: 追加药物靶基因 (用于 aissa 伪扰动增强的嵌入)
DRUG_TARGETS = {"osimertinib": ["EGFR"], "crizotinib": ["ALK", "ROS1", "MET"]}
extra_drug = [t for ts in DRUG_TARGETS.values() for t in ts
              if t in X.columns and t not in genes]
genes += extra_drug
g2i = {g: i for i, g in enumerate(genes)}
print(f"genes: {len(genes)} (HVG {int(hv.sum())} + targets {len(extra)}"
      f" + drug {extra_drug})", flush=True)

ctrl_X = X.loc[is_ctrl, genes].values
ctrl_mean = ctrl_X.mean(0)
gene_cells = {g: X.index[(tgt == g).values] for g in all_targets}
genes_use = [g for g in all_targets
             if len(gene_cells[g]) >= MIN_CELLS and g in g2i]
print(f"controls: {ctrl_X.shape[0]}; usable targets (>= {MIN_CELLS} cells): "
      f"{len(genes_use)}", flush=True)

# 嵌入 (升级: 共表达 PCA16 + PPR 平滑调控子 PCA8 + 截距)
E_co = coexpr_embedding(ctrl_X)
E_re = regulon_embedding_smoothed(genes, g2i)
E_feat = np.hstack([E_co, E_re])
E_feat = E_feat / (np.linalg.norm(E_feat, axis=1, keepdims=True) + 1e-9)
E = np.hstack([E_feat, np.ones((len(genes), 1))])
print(f"embedding: co {E_co.shape[1]} + re-smoothed {E_re.shape[1]} "
      f"+ intercept = {E.shape[1]}", flush=True)

delta_true = {g: X.loc[gene_cells[g], genes].values.mean(0) - ctrl_mean
              for g in genes_use}

# 与 v5 完全相同的 seed-0 划分
perm = rng.permutation(len(genes_use))
n_test = max(3, len(genes_use) // 3)
test_idx = set(perm[:n_test].tolist())
train_genes = [g for i, g in enumerate(genes_use) if i not in test_idx]
test_genes = [g for i, g in enumerate(genes_use) if i in test_idx]
print(f"LOO split: train={len(train_genes)} test={len(test_genes)}", flush=True)
print("test genes:", test_genes, flush=True)
mean_delta = np.vstack([delta_true[g] for g in train_genes]).mean(0)

# 基因-基因共表达相似度 (NW/核用): 对照矩阵列向量的相关
Xc = ctrl_X - ctrl_X.mean(0)
sd = Xc.std(0)
sd[sd < 1e-9] = 1.0
Xs = Xc / sd
G = (Xs.T @ Xs) / ctrl_X.shape[0]  # genes x genes 相关矩阵

# ================= aissa 单药伪扰动 (映射进 dat17 空间) =================
print("加载 aissa 做单药增强 ...", flush=True)
ai = sc.read_h5ad(f"{CORE}/aissa_processed.h5ad")
Xi = ai.raw.X.toarray() if hasattr(ai.raw.X, "toarray") else np.asarray(ai.raw.X)
Xi = pd.DataFrame(Xi, index=ai.obs_names, columns=ai.raw.var_names)
cond = ai.obs["perturbation"].astype(str).str.strip().str.lower()
cmap_cells = {c: Xi.index[cond == c] for c in cond.unique()}
shared = [g for g in genes if g in Xi.columns]
print(f"aissa 共享基因: {len(shared)}/{len(genes)}", flush=True)
aug_rows = []   # (gene, delta_in_dat17_space)
for drug, tgts in DRUG_TARGETS.items():
    if drug not in cmap_cells or len(cmap_cells[drug]) < 50:
        continue
    dvec = Xi.loc[cmap_cells[drug], shared].values.mean(0) \
        - Xi.loc[cmap_cells["control"], shared].values.mean(0)
    full = np.zeros(len(genes))
    gidx = [g2i[g] for g in shared]
    full[gidx] = dvec
    for t in tgts:
        if t in g2i:
            aug_rows.append((t, full))
            print(f"  aug: {drug} -> {t} (|delta|_mean="
                  f"{np.abs(dvec).mean():.3f})", flush=True)

# ================= 算子定义 =================
def nw_predict(sim_row, tr_idx, D_tr, tau):
    """Nadaraya-Watson: 归一化指数权重插值"""
    w = np.exp(sim_row[tr_idx] / max(tau, 1e-6))
    w = w / (w.sum() + 1e-9)
    return w @ D_tr


def knn_predict(sim_row, tr_idx, D_tr, k):
    order = np.argsort(sim_row[tr_idx])[::-1][:k]
    w = sim_row[tr_idx][order]
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    return w @ D_tr[order]


def rbf_kr_fit_predict(E_tr, D_tr, E_te, tau, lam):
    K = E_tr @ E_tr.T
    K = np.exp(-(np.maximum(0, 2 - 2 * K)) / (2 * tau ** 2))
    kte = E_te @ E_tr.T
    kte = np.exp(-(np.maximum(0, 2 - 2 * kte)) / (2 * tau ** 2))
    A = K + lam * np.eye(K.shape[0])
    coef = np.linalg.solve(A, D_tr)
    return kte @ coef


def loo_operator_fit(E, D, idx_all, op_kwargs):
    """在给定基因集合内做留一, 返回每个基因的 LOO 预测 (算子选择用)"""
    preds = np.zeros_like(D)
    for i in range(len(idx_all)):
        tr = [j for j in range(len(idx_all)) if j != i]
        tr_idx = [idx_all[j] for j in tr]
        te_idx = idx_all[i]
        preds[i] = predict_one(op_kwargs, tr_idx, D[tr], te_idx)
    return preds


def predict_one(op, tr_idx, D_tr, te_idx):
    kind = op["kind"]
    if kind == "nw_emb":
        sim = E[:, :-1] @ E[te_idx, :-1] / (
            np.linalg.norm(E[:, :-1], axis=1) *
            np.linalg.norm(E[te_idx, :-1]) + 1e-9)
        return nw_predict(sim, tr_idx, D_tr, op["tau"])
    if kind == "nw_corr":
        mode = op["sim"]
        row = G[te_idx]
        if mode == "abs":
            row = np.abs(row)
        elif mode == "pos":
            row = np.clip(row, 0, None)
        return nw_predict(row, tr_idx, D_tr, op["tau"])
    if kind == "knn_corr":
        row = G[te_idx]
        return knn_predict(np.clip(row, 0, None), tr_idx, D_tr, op["k"])
    if kind == "rbf_kr":
        return rbf_kr_fit_predict(E[tr_idx], D_tr,
                                 E[[te_idx]], op["tau"], op["lam"])[0]
    raise ValueError(kind)


def eval_delta(pred, true):
    scale = np.abs(true).mean() + 1e-9
    return dict(delta_pearson=safe_corr(pred, true),
                delta_mae_norm=np.abs(pred - true).mean() / scale,
                deg_top50_overlap=len(
                    set(np.argsort(np.abs(true))[-50:]) &
                    set(np.argsort(np.abs(pred))[-50:])) / 50)


# ================= 5 折 CV 选模型 (27 可用基因) =================
print("\n=== 5-fold CV over 27 genes ===", flush=True)
all_idx = [g2i[g] for g in genes_use]
D_all = np.vstack([delta_true[g] for g in genes_use])
kf = np.array_split(rng.permutation(len(genes_use)), 5)

operator_grid = []
for tau in [0.02, 0.05, 0.1, 0.2, 0.4]:
    operator_grid.append({"kind": "nw_emb", "tau": tau})
for sim in ["corr", "abs", "pos"]:
    for tau in [0.02, 0.05, 0.1, 0.2, 0.4]:
        operator_grid.append({"kind": "nw_corr", "sim": sim, "tau": tau})
for k in [3, 5, 8]:
    operator_grid.append({"kind": "knn_corr", "k": k})
for tau in [0.5, 1.0, 2.0]:
    for lam in [1.0, 10.0, 100.0]:
        operator_grid.append({"kind": "rbf_kr", "tau": tau, "lam": lam})

cv_rows = []
for op in operator_grid:
    rs, maes, ovs = [], [], []
    for fold in kf:
        te = fold.tolist()
        tr = [i for i in range(len(genes_use)) if i not in set(te)]
        tr_idx = [all_idx[j] for j in tr]
        te_idx_l = [all_idx[j] for j in te]
        D_tr = D_all[tr]
        for j, ti in zip(te, te_idx_l):
            p = predict_one(op, tr_idx, D_tr, ti)
            m = eval_delta(p, D_all[j])
            rs.append(m["delta_pearson"]); maes.append(m["delta_mae_norm"])
            ovs.append(m["deg_top50_overlap"])
    cv_rows.append(dict(op=json.dumps(op), cv_pearson=np.nanmean(rs),
                        cv_mae=np.nanmean(maes), cv_deg=np.nanmean(ovs)))
    print(f"  {op} -> r={np.nanmean(rs):.4f}", flush=True)
cv = pd.DataFrame(cv_rows).sort_values("cv_pearson", ascending=False)
cv.to_csv(f"{OUT}/v6_cv.csv", index=False)
print(cv.head(8), flush=True)

# 训练集内 LOO 选 NW tau / k / 混合系数, 供固定测试集使用
print("\n=== train-set LOO for hyper-params ===", flush=True)
tr_g2i = [g2i[g] for g in train_genes]
D_tr = np.vstack([delta_true[g] for g in train_genes])

best_op = None
for op in operator_grid:
    loo_pred = np.zeros_like(D_tr)
    for i in range(len(tr_g2i)):
        tr_idx = [t for j, t in enumerate(tr_g2i) if j != i]
        loo_pred[i] = predict_one(op, tr_idx,
                                  np.delete(D_tr, i, axis=0), tr_g2i[i])
    r = np.nanmean([safe_corr(loo_pred[i], D_tr[i]) for i in range(len(D_tr))])
    op["train_loo_r"] = r
    if best_op is None or r > best_op["train_loo_r"]:
        best_op = op
print("best operator (train LOO):", best_op, flush=True)

# 混合系数 alpha (模型 + MeanDelta) 用训练集 LOO 选
loo_pred_best = np.zeros_like(D_tr)
for i in range(len(tr_g2i)):
    tr_idx = [t for j, t in enumerate(tr_g2i) if j != i]
    loo_pred_best[i] = predict_one(best_op, tr_idx,
                                   np.delete(D_tr, i, axis=0), tr_g2i[i])
m_tr = D_tr.mean(0)
alphas = np.arange(0, 1.01, 0.05)
a_scores = [np.nanmean([safe_corr(a * loo_pred_best[i] + (1 - a) * m_tr,
                                  D_tr[i]) for i in range(len(D_tr))])
            for a in alphas]
best_alpha = float(alphas[int(np.nanargmax(a_scores))])
print(f"best alpha={best_alpha} (train-LOO r={max(a_scores):.4f})", flush=True)

# 增强版: 训练集 + aissa 伪扰动行
tr_aug_idx = tr_g2i + [g2i[t] for t, _ in aug_rows]
D_aug = np.vstack([D_tr] + [d[None, :] for _, d in aug_rows])
loo_pred_aug = np.zeros_like(D_tr)
for i in range(len(tr_g2i)):
    keep = [j for j in range(len(tr_aug_idx)) if tr_aug_idx[j] != tr_g2i[i]]
    loo_pred_aug[i] = predict_one(best_op, [tr_aug_idx[j] for j in keep],
                                  D_aug[keep], tr_g2i[i])
r_aug = np.nanmean([safe_corr(loo_pred_aug[i], D_tr[i])
                    for i in range(len(D_tr))])
a_scores_aug = [np.nanmean([safe_corr(a * loo_pred_aug[i] + (1 - a) * m_tr,
                                      D_tr[i])
                            for i in range(len(D_tr))]) for a in alphas]
best_alpha_aug = float(alphas[int(np.nanargmax(a_scores_aug))])
print(f"augmented: train-LOO r={r_aug:.4f}, alpha_aug={best_alpha_aug}", flush=True)

# ================= 固定测试集评估 =================
print("\n=== fixed test-set evaluation ===", flush=True)
rows, pred_store = [], {}
for gn in test_genes:
    te = g2i[gn]
    true = delta_true[gn]
    # v5 ridge 复现
    W, lam, _ = ridge_fit(E[tr_g2i], D_tr)
    pred_ridge = (E[[te]] @ W)[0]
    pred_op = predict_one(best_op, tr_g2i, D_tr, te)
    pred_op_aug = predict_one(best_op, tr_aug_idx, D_aug, te)
    cands = {
        "Ridge-v5": pred_ridge,
        "BestOp": pred_op,
        "BestOp-aug": pred_op_aug,
        f"Blend-a{best_alpha}": best_alpha * pred_op + (1 - best_alpha) * m_tr,
        f"BlendAug-a{best_alpha_aug}":
            best_alpha_aug * pred_op_aug + (1 - best_alpha_aug) * m_tr,
        "MeanDelta": m_tr,
        "Identity": np.zeros_like(true),
    }
    for tag, p in cands.items():
        m = eval_delta(p, true)
        rows.append(dict(perturbation=gn, model=tag, **m))
        pred_store.setdefault(tag, {})[gn] = p
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/v6_test_benchmark.csv", index=False)
summ = res.groupby("model")[["delta_pearson", "delta_mae_norm",
                             "deg_top50_overlap"]].mean()
summ.to_csv(f"{OUT}/v6_test_summary.csv")
print(summ, flush=True)

# 置换零模型 (best blend)
null_rows = []
for gn in test_genes:
    true = delta_true[gn]
    cs = []
    for _ in range(100):
        ri = rng.randint(0, len(E))
        while ri == g2i[gn]:
            ri = rng.randint(0, len(E))
        e_row = E[[ri]].copy()
        e_row[0, -1] = 1.0
        Wn, _, _ = ridge_fit(E[tr_g2i], D_tr)
        cs.append(safe_corr((e_row @ Wn)[0], true))
    null_rows.append(dict(perturbation=gn, model="PermutedNull",
                          delta_pearson=np.nanmean(cs),
                          delta_mae_norm=np.nan, deg_top50_overlap=np.nan))
pd.DataFrame(null_rows).to_csv(f"{OUT}/v6_permuted_null.csv", index=False)

# VAE (per-cell pearson for best blend)
vae = train_vae(ctrl_X, 25, "vae-v6")
xt_store = {}
for gn in test_genes:
    xt = X.loc[gene_cells[gn], genes].values
    pred = pred_store[f"Blend-a{best_alpha}"][gn]
    xhat = ctrl_X[rng.choice(len(ctrl_X), min(1200, len(ctrl_X)),
                             replace=False)] + pred
    pc = np.mean([pearsonr(xhat[i], xt[i % len(xt)])[0]
                  for i in range(len(xhat))])
    xt_store[gn] = pc
    print(f"per-cell pearson {gn}: {pc:.4f}", flush=True)
pd.DataFrame({"perturbation": list(xt_store),
              "percell_pearson": list(xt_store.values())}).to_csv(
    f"{OUT}/v6_percell.csv", index=False)

np.savez(f"{OUT}/v6_operator.npz", E=E, genes=np.array(genes),
         best_op=json.dumps(best_op), alpha=best_alpha,
         alpha_aug=best_alpha_aug, test_genes=np.array(test_genes),
         train_genes=np.array(train_genes))
torch.save(vae.state_dict(), f"{OUT}/v6_vae.pt")

print(f"\nV6 total time: {(time.time()-t0)/60:.1f} min")
print("MAGWORLD_V6_DONE")
