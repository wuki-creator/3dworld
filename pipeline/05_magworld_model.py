# -*- coding: utf-8 -*-
"""MAGWorld: 智能体引导的虚拟细胞世界模型 v3

核心架构:
  World VAE 学习对照细胞状态流形 (lat=32);
  基因嵌入 E (对照细胞基因共表达谱 PCA) + 线性扰动算子 W (岭回归):
      delta_hat(gene) = W @ E[gene]
  对未见靶基因/未见药物组合零样本预测扰动效应谱。

实验一: Datlinger2017 CRISPRi (33 靶基因) 基因级留一
  - 指标: delta_pearson / delta_mae_norm / DEG top-50 / 判别排名 / 单细胞相关
  - 基线: MeanDelta (训练集平均效应), Identity (无效应)
实验二: Aissa2021 药物组合跨数据集 in silico 预测
  - 用 dat17 学到的 W 与 aissa 共享基因空间, 由组合靶点嵌入均值预测
    osimertinib+crizotinib 联合效应, 与真实双药 / 观测加性模型比较
  - 协同度: 模型预测 vs 观测
实验三: pbmc3k in silico TF KO + CollecTRI 活性重评分 (07 脚本)
"""
import os
import json
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
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld"
os.makedirs(OUT, exist_ok=True)
EMB_DIM = 16
MIN_CELLS = 60
rng = np.random.RandomState(0)


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

    def decode(self, z):
        return self.dec(z)


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
    """对照细胞基因共表达谱 PCA -> 基因嵌入"""
    Xc = Xctrl - Xctrl.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    E = Vt[:dim].T * S[:dim]  # genes x dim
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return E


def regulon_embedding(genes, g2i, dim=8):
    """CollecTRI 调控子谱 PCA -> 基因嵌入 (机制先验)"""
    import pandas as pd
    net = pd.read_csv(
        "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
    )[["source", "target", "weight"]].dropna()
    tfs = sorted(net.source.unique())
    tf2i = {t: i for i, t in enumerate(tfs)}
    M = np.zeros((len(genes), len(tfs)))
    for s, tg, w in net.itertuples(index=False):
        if tg in g2i and s in tf2i:
            M[g2i[tg], tf2i[s]] += w
    if M.std() == 0:
        return np.zeros((len(genes), dim))
    U, S, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    k = min(dim, Vt.shape[0])
    R = U[:, :k] * S[:k]
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    return R


def ridge_fit(E_tr, D_tr, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)):
    """LOO 选岭参数; 返回 W (dim x genes) 与最优 lam"""
    n = E_tr.shape[0]
    best = (None, None, np.inf)
    I = np.eye(E_tr.shape[1])
    for lam in lams:
        # LOO 预测 (ridge 留一公式)
        A = E_tr @ E_tr.T + lam * np.eye(n)
        Ainv = np.linalg.inv(A)
        H = E_tr @ E_tr.T @ Ainv
        loo_pred = (H @ D_tr - np.diag(H)[:, None] * D_tr) / \
            (1 - np.diag(H))[:, None]
        err = ((loo_pred - D_tr) ** 2).mean()
        if err < best[2]:
            W = np.linalg.solve(E_tr.T @ E_tr + lam * I, E_tr.T @ D_tr)
            best = (W, lam, err)
    return best


# ================= 实验一: Datlinger CRISPRi 基因级留一 =================
print("加载 Datlinger2017 (dat17) ...", flush=True)
ad = sc.read_h5ad(f"{CORE}/dat17_processed.h5ad")
X = ad.raw.X.toarray() if hasattr(ad.raw.X, "toarray") else np.asarray(ad.raw.X)
X = pd.DataFrame(X, index=ad.obs_names, columns=ad.raw.var_names)
hv = ad.var["highly_variable"].reindex(X.columns).fillna(False).values
genes = list(X.columns[hv])

tgt = ad.obs["target"].astype(str)
tgt = tgt.where(tgt.notna() & (tgt.str.lower() != "nan"), "")
pert = ad.obs["perturbation"].astype(str)
is_ctrl = (tgt == "") | pert.str.lower().isin(
    ["control", "ctrl", "nt", "non-targeting"])

all_targets = sorted(set(t for t in tgt if t))
extra = [t for t in all_targets if t in X.columns and t not in genes]
genes += extra
g2i = {g: i for i, g in enumerate(genes)}
print(f"genes: {len(genes)} (HVG {len(genes)-len(extra)} + targets {len(extra)})",
      flush=True)

ctrl_X = X.loc[is_ctrl, genes].values
ctrl_mean = ctrl_X.mean(0)
gene_cells = {g: X.index[(tgt == g).values] for g in all_targets}
genes_use = [g for g in all_targets
             if len(gene_cells[g]) >= MIN_CELLS and g in g2i]
print(f"controls: {ctrl_X.shape[0]}; usable target genes: {len(genes_use)}",
      flush=True)

# World VAE (对照流形; 用于单细胞残差重构)
vae = train_vae(ctrl_X, 30, "vae-dat17")

# 基因嵌入: 共表达谱 PCA (16) ⊕ CollecTRI 调控子谱 PCA (8) + 真实效应谱
E_co = coexpr_embedding(ctrl_X)
E_re = regulon_embedding(genes, g2i)
E_feat = np.hstack([E_co, E_re])
E_feat = E_feat / (np.linalg.norm(E_feat, axis=1, keepdims=True) + 1e-9)
E = np.hstack([E_feat, np.ones((len(genes), 1))])  # 截距: 通用扰动响应分量
print(f"gene embedding: coexpr {E_co.shape[1]} + regulon {E_re.shape[1]}"
      f" = {E.shape[1]} dims", flush=True)
delta_true = {g: X.loc[gene_cells[g], genes].values.mean(0) - ctrl_mean
              for g in genes_use}

perm = rng.permutation(len(genes_use))
n_test = max(3, len(genes_use) // 3)
test_idx = set(perm[:n_test].tolist())
train_genes = [g for i, g in enumerate(genes_use) if i not in test_idx]
test_genes = [g for i, g in enumerate(genes_use) if i in test_idx]
print(f"gene-level LOO: train={len(train_genes)} test={len(test_genes)}",
      flush=True)
print("test genes:", test_genes, flush=True)

E_tr = E[[g2i[g] for g in train_genes]]
D_tr = np.vstack([delta_true[g] for g in train_genes])
W, lam, loo_err = ridge_fit(E_tr, D_tr)
print(f"ridge lam={lam} loo_mse={loo_err:.5f}", flush=True)

train_delta_mean = D_tr.mean(0)

rows, all_pred_delta, all_true_delta = [], {}, {}
for gn in test_genes:
    true_delta = delta_true[gn]
    pred_delta = (E[[g2i[gn]]] @ W)[0]
    # 嵌入置换零模型 (100 次, 仅置换特征部分, 保留截距): 检验基因特异结构
    null_c = []
    for _ in range(100):
        ei = g2i[gn]
        ri = rng.randint(0, len(E))
        while ri == ei:
            ri = rng.randint(0, len(E))
        e_row = E[[ri]].copy()
        e_row[0, -1] = 1.0
        null_c.append(safe_corr((e_row @ W)[0], true_delta))
    null_mean = np.nanmean(null_c)
    models = {"MAGWorld": pred_delta,
              "MeanDelta": train_delta_mean,
              "Identity": np.zeros_like(true_delta),
              "PermutedNull": None}
    # 单细胞预测 = 对照细胞 + 预测效应 (scGen 式加性)
    xt = X.loc[gene_cells[gn], genes].values
    xhat = ctrl_X[rng.choice(len(ctrl_X), min(1500, len(ctrl_X)),
                             replace=False)] + pred_delta
    for tag, pdlt in models.items():
        if tag == "PermutedNull":
            rows.append(dict(perturbation=gn, model=tag,
                             delta_pearson=null_mean, delta_mae_norm=np.nan,
                             deg_top50_overlap=np.nan, percell_pearson=np.nan))
            continue
        r_d = safe_corr(pdlt, true_delta)
        scale = np.abs(true_delta).mean() + 1e-9
        dmae = np.abs(pdlt - true_delta).mean() / scale
        ov = len(set(np.argsort(np.abs(true_delta))[-50:]) &
                 set(np.argsort(np.abs(pdlt))[-50:])) / 50
        row = dict(perturbation=gn, model=tag, delta_pearson=r_d,
                   delta_mae_norm=dmae, deg_top50_overlap=ov)
        if tag == "MAGWorld":
            row["percell_pearson"] = np.mean(
                [pearsonr(xhat[i], xt[i % len(xt)])[0]
                 for i in range(len(xhat))])
        else:
            row["percell_pearson"] = np.nan
        rows.append(row)
    all_pred_delta[gn] = pred_delta
    all_true_delta[gn] = true_delta
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/loo_benchmark.csv", index=False)


def disc_rank(pred_d, true_d):
    Ps = list(pred_d)
    nr = []
    for i, p in enumerate(Ps):
        sims = []
        for q in Ps:
            a, b = pred_d[p], true_d[q]
            sims.append(np.dot(a, b) / (np.linalg.norm(a) *
                                        np.linalg.norm(b) + 1e-9))
        nr.append(int(np.argsort(sims)[::-1].tolist().index(i)) + 1)
    return nr


disc = disc_rank(all_pred_delta, all_true_delta)
pd.DataFrame({"perturbation": list(all_pred_delta),
              "cosine_rank": disc}).to_csv(
    f"{OUT}/perturbation_discrimination.csv", index=False)

# in silico 剂量反应 (效应线性缩放)
dose_grid = np.linspace(0, 2.0, 9)
dose_curves, dose_names = {}, []
for gn in (test_genes + train_genes[:3]):
    pdlt = all_pred_delta.get(gn)
    if pdlt is None:
        pdlt = (E[[g2i[gn]]] @ W)[0]
    curve = np.vstack([ctrl_mean + a * pdlt for a in dose_grid])
    dose_curves[f"curve_{len(dose_curves)}"] = curve
    dose_names.append(gn)
np.savez(f"{OUT}/dose_response.npz", dose=dose_grid,
         genes=np.array(genes), **dose_curves)
with open(f"{OUT}/dose_response_perts.json", "w") as f:
    json.dump(dose_names, f)

# ================= 实验二: Aissa 药物组合 in silico =================
print("加载 Aissa (药物组合) ...", flush=True)
ai = sc.read_h5ad(f"{CORE}/aissa_processed.h5ad")
Xi = ai.raw.X.toarray() if hasattr(ai.raw.X, "toarray") else np.asarray(ai.raw.X)
Xi = pd.DataFrame(Xi, index=ai.obs_names, columns=ai.raw.var_names)
hvi = ai.var["highly_variable"].reindex(Xi.columns).fillna(False).values
genes_i = list(Xi.columns[hvi])
cond = ai.obs["perturbation"].astype(str).str.strip().str.lower()
cmap_cells = {c: Xi.index[cond == c] for c in cond.unique()}
ctrl_i = Xi.loc[cmap_cells["control"], genes_i].values
ctrl_mean_i = ctrl_i.mean(0)

DRUG_TARGETS = {"osimertinib": ["EGFR"], "crizotinib": ["ALK", "ROS1",
                                                        "MET"]}
extra_i = [t for ts in DRUG_TARGETS.values() for t in ts
           if t in Xi.columns and t not in genes_i]
genes_i += extra_i
ctrl_i = Xi.loc[cmap_cells["control"], genes_i].values
ctrl_mean_i = ctrl_i.mean(0)
print(f"Aissa genes: {len(genes_i)} (extra targets: {extra_i})", flush=True)

# 药物嵌入 = 靶基因在 aissa 对照共表达空间的嵌入均值
Ei = coexpr_embedding(ctrl_i)
g2i_i = {gname: i for i, gname in enumerate(genes_i)}


class ShiftNet(nn.Module):
    def __init__(self, n_genes, emb=EMB_DIM, lat=32):
        super().__init__()
        self.emb = nn.Embedding(n_genes, emb)
        self.net = nn.Sequential(nn.Linear(lat + emb, 64), nn.ReLU(),
                                 nn.Linear(64, lat))

    def forward(self, z, gene_idx):
        e = self.emb(gene_idx).mean(1)
        return self.net(torch.cat([z, e], 1))


vae_i = train_vae(ctrl_i, 25, "vae-aissa")
with torch.no_grad():
    zc_i_all = vae_i.encode(torch.tensor(ctrl_i, dtype=torch.float32))
    center_i = zc_i_all.mean(0, keepdim=True)
zc_i = (zc_i_all - center_i).numpy()

mean_drug = {d: Xi.loc[cmap_cells[d], genes_i].values.mean(0)
             for d in DRUG_TARGETS if d in cmap_cells
             and len(cmap_cells[d]) >= 50}

g_i = ShiftNet(len(genes_i))
with torch.no_grad():
    for gn, idx in g2i_i.items():
        if idx < Ei.shape[0]:
            v = torch.tensor(Ei[idx], dtype=torch.float32)
            v = v / (v.norm() + 1e-9) * 0.5
            g_i.emb.weight[idx] = v
opt_i = torch.optim.Adam(g_i.parameters(), lr=1e-3)
rng_i = np.random.RandomState(0)
for ep in range(60):
    tot, nb = 0.0, 0
    for drug, tgts in DRUG_TARGETS.items():
        if drug not in mean_drug:
            continue
        n_c = 512
        zc_b = torch.tensor(zc_i[rng_i.choice(len(zc_i), n_c, replace=False)],
                            dtype=torch.float32)
        gi = torch.tensor([g2i_i[t] for t in tgts if t in g2i_i],
                          dtype=torch.long)
        if len(gi) == 0:
            continue
        gi = gi.unsqueeze(0).repeat(n_c, 1)
        xhat = vae_i.decode(zc_b + g_i(zc_b, gi))
        loss = nn.functional.mse_loss(
            xhat, torch.tensor(mean_drug[drug], dtype=torch.float32))
        opt_i.zero_grad(); loss.backward(); opt_i.step()
        tot += loss.item(); nb += 1
    if (ep + 1) % 20 == 0:
        print(f"  [shift-aissa] {ep+1}/60 mse={tot/max(nb,1):.5f}",
              flush=True)

vae_i.eval(); g_i.eval()

idx = rng_i.choice(len(zc_i), min(1500, len(zc_i)), replace=False)
zb = torch.tensor(zc_i[idx], dtype=torch.float32)


def pred_drug_delta(tgts):
    gi = torch.tensor([g2i_i[t] for t in tgts if t in g2i_i],
                      dtype=torch.long)
    if len(gi) == 0:
        return None
    gi = gi.unsqueeze(0).repeat(len(idx), 1)
    with torch.no_grad():
        x = vae_i.decode(zb + g_i(zb, gi)).numpy()
    return x.mean(0) - ctrl_mean_i


combo_rows = []
combo_key = "osimertinib+crizotinib"
if combo_key in cmap_cells and len(cmap_cells[combo_key]) >= 50:
    xt_c = Xi.loc[cmap_cells[combo_key], genes_i].values
    true_combo_delta = xt_c.mean(0) - ctrl_mean_i
    d_combo = pred_drug_delta(DRUG_TARGETS["osimertinib"] +
                              DRUG_TARGETS["crizotinib"])
    d_osa = pred_drug_delta(DRUG_TARGETS["osimertinib"])
    d_cri = pred_drug_delta(DRUG_TARGETS["crizotinib"])
    d_osa_obs = mean_drug["osimertinib"] - ctrl_mean_i
    d_cri_obs = mean_drug["crizotinib"] - ctrl_mean_i
    d_add_obs = d_osa_obs + d_cri_obs
    scale_c = np.abs(true_combo_delta).mean() + 1e-9
    for tag, pdlt in [("MAGWorld-combo", d_combo),
                      ("Additivity", d_add_obs),
                      ("Identity", np.zeros_like(true_combo_delta))]:
        if pdlt is None:
            continue
        ov = len(set(np.argsort(np.abs(true_combo_delta))[-50:]) &
                 set(np.argsort(np.abs(pdlt))[-50:])) / 50
        combo_rows.append(dict(
            model=tag,
            delta_pearson=safe_corr(pdlt, true_combo_delta),
            delta_mae_norm=np.abs(pdlt - true_combo_delta).mean() / scale_c,
            deg_top50_overlap=ov))

    def syn_of(dc, d1, d2):
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        return abs(np.linalg.norm(dc) - n1 - n2) / (n1 + n2 + 1e-9)
    if d_combo is not None and d_osa is not None and d_cri is not None:
        syn_model = syn_of(d_combo, d_osa, d_cri)
        syn_obs = syn_of(true_combo_delta, d_osa_obs, d_cri_obs)
        pd.DataFrame(
            {"source": ["MAGWorld prediction", "Observed data"],
             "synergy": [syn_model, syn_obs]}).to_csv(
            f"{OUT}/combinatorial_synergy.csv", index=False)
        print(f"synergy: model={syn_model:.3f} observed={syn_obs:.3f}",
              flush=True)
pd.DataFrame(combo_rows).to_csv(f"{OUT}/drug_combo_benchmark.csv",
                                index=False)
print(pd.DataFrame(combo_rows), flush=True)

np.savez(f"{OUT}/magworld_operator.npz", W=W, E=E, genes=np.array(genes),
         lam=lam)
torch.save({"vae_aissa": vae_i.state_dict(), "g_aissa": g_i.state_dict(),
            "center_aissa": center_i, "genes_aissa": genes_i},
           f"{OUT}/magworld_model.pt")
summ = res.groupby("model")[["delta_pearson", "delta_mae_norm",
                             "deg_top50_overlap"]].mean()
summ.to_csv(f"{OUT}/benchmark_summary.csv")
print(summ, flush=True)
print("MAGWORLD_DONE")
