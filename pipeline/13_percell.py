# -*- coding: utf-8 -*-
"""MAGWorld v9: scGen 类条件 VAE 单细胞扰动预测 (seen-perturbation 协议)。

动机: v6/v7/v8 已用严格零假设证明未见基因的 delta 特异分量不可预测
(specific_pearson≈0, 判别排名=随机, 2000 训练基因亦然)。社区标准协议
(scGen/CPA) 的正面战场是 seen-perturbation 单细胞预测: 同一扰动的细胞
留出, 用对照细胞 + 扰动条件生成预测细胞。

模型: VAE (对照+训练扰动细胞) + ShiftNet(z, e_g) -> delta-z,
      e_g 初始化为 CollecTRI 嵌入 (可迁移到未见基因, 零样本路径同测)。
评估: 每个训练基因 20% 细胞留出
  - 基线1 MeanDelta-shift: ctrl 细胞 + 该基因训练均值 delta
  - 基线2 GlobalMean-shift: ctrl 细胞 + 全局均值 delta
  - 基线3 Identity: ctrl 细胞
  - MAGWorld-cVAE: decode(z_ctrl + ShiftNet(z_ctrl, e_g))
  指标: per-cell pearson (随机配对), 预测 delta vs 留出 delta 的
        pearson / specific / 幅度误差 / DEG top50
  - 零样本: 对 34 个测试基因直接生成, 报告 delta 指标 (对照 MeanDelta)
输出: results/magworld/percell/
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
from scipy import sparse
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)
rng = np.random.RandomState(0)
t0 = time.time()

H5AD = "/nfs_beijing/zizhuo/vcc/data/norman/NormanWeissman2019_filtered.h5ad"
PREP = "/nfs_beijing/zizhuo/vcc/results/magworld/norman/norman_prep.npz"
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/percell"
os.makedirs(OUT, exist_ok=True)
EPOCHS = 30
HOLD_FRAC = 0.2


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return pearsonr(a, b)[0]


# ================= 1. 数据 =================
print("加载 Norman ...", flush=True)
P = np.load(PREP, allow_pickle=True)
genes = P["genes"].astype(str).tolist()
g2i = {g: i for i, g in enumerate(genes)}
E_re = P["E_re"]
m_tr = P["m_tr"]
train_genes = P["train_genes"].astype(str).tolist()
test_genes = P["test_genes"].astype(str).tolist()

ad = sc.read_h5ad(H5AD)
X = ad.X.tocsr() if sparse.isspmatrix(ad.X) else sparse.csr_matrix(ad.X)
per = ad.obs["perturbation"].astype(str).values
nperts = ad.obs["nperts"].astype(int).values
raw2i = {g: i for i, g in enumerate(ad.var_names.astype(str))}
cols = np.array([raw2i[g] for g in genes if g in raw2i])
keep_genes_mask = np.array([g in raw2i for g in genes])
genes_k = [g for g in genes if g in raw2i]
print(f"genes usable: {len(genes_k)}/{len(genes)}", flush=True)

ctrl_sel = np.where((per == "control") | (nperts == 0))[0]
ctrl_X = np.asarray(X[ctrl_sel][:, cols].todense(), dtype=np.float32)
print(f"ctrl cells: {ctrl_X.shape}", flush=True)

# 每个训练基因: 80% 训练 / 20% 留出
tr_cells, ho_cells, tr_delta = {}, {}, {}
for g in train_genes:
    if g not in genes_k:
        continue
    sel = np.where((per == g) & (nperts == 1))[0]
    if len(sel) < 50:
        continue
    perm = rng.permutation(len(sel))
    n_ho = max(20, int(len(sel) * HOLD_FRAC))
    ho = sel[perm[:n_ho]]
    tr = sel[perm[n_ho:]]
    tr_cells[g] = tr
    ho_cells[g] = ho
    Xtr = np.asarray(X[tr][:, cols].todense(), dtype=np.float32)
    tr_delta[g] = Xtr.mean(0) - ctrl_X.mean(0)
use_genes = sorted(tr_cells)
print(f"usable train genes: {len(use_genes)}", flush=True)

X_tr_pert = np.vstack([np.asarray(X[tr_cells[g]][:, cols].todense(),
                                 dtype=np.float32) for g in use_genes])
gene_lab = np.concatenate([np.full(len(tr_cells[g]), i)
                           for i, g in enumerate(use_genes)])
X_all = np.vstack([ctrl_X, X_tr_pert])
cond_lab = np.concatenate([np.full(len(ctrl_X), -1), gene_lab])
print(f"training pool: {X_all.shape}", flush=True)

# ================= 2. 模型 =================
class VAE(nn.Module):
    def __init__(self, n_in, lat=32, hid=256):
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


class ShiftNet(nn.Module):
    def __init__(self, n_genes, emb, lat=32):
        super().__init__()
        self.emb = nn.Embedding(n_genes + 1, emb)  # +1: null condition
        self.net = nn.Sequential(nn.Linear(lat + emb, 128), nn.ReLU(),
                                 nn.Linear(128, lat))

    def forward(self, z, gene_idx):
        e = self.emb(gene_idx)
        return self.net(torch.cat([z, e], 1))


n_g = len(genes_k)
gidx_k = {g: i for i, g in enumerate(genes_k)}
vae = VAE(X_all.shape[1])
shift = ShiftNet(n_g, E_re.shape[1])
with torch.no_grad():
    for gname, gi in gidx_k.items():
        if gname in g2i:
            v = torch.tensor(E_re[g2i[gname]], dtype=torch.float32)
            shift.emb.weight[gi] = v / (v.norm() + 1e-9) * 0.5
    shift.emb.weight[n_g].zero_()  # null (ctrl)

X_t = torch.tensor(X_all)
cond_t = torch.tensor(
    [gidx_k[use_genes[i]] if i >= 0 else n_g for i in cond_lab],
    dtype=torch.long)
opt = torch.optim.Adam(
    list(vae.parameters()) + list(shift.parameters()), lr=1e-3)
n = X_t.shape[0]
for ep in range(EPOCHS):
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, 2048):
        idx = perm[i:i + 2048]
        xb, cb = X_t[idx], cond_t[idx]
        xr, mu, lv = vae(xb)
        dz = shift(mu, cb)
        xh = vae.dec(mu + dz)
        rec0 = nn.functional.mse_loss(xr, xb)
        rec1 = nn.functional.mse_loss(xh, xb)
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        loss = rec1 + 0.1 * rec0 + 1e-3 * kl
        opt.zero_grad(); loss.backward(); opt.step()
        tot += rec1.item() * len(idx)
    if (ep + 1) % 5 == 0:
        print(f"  [cvae] {ep+1}/{EPOCHS} recon={tot/n:.4f}", flush=True)
vae.eval(); shift.eval()

with torch.no_grad():
    z_ctrl = vae.encode(torch.tensor(ctrl_X)).numpy()
ctrl_mean = ctrl_X.mean(0)

# ================= 3. 留出评估 =================
print("\n=== held-out per-cell evaluation ===", flush=True)
N_PICK = 1200
rows = []
ctrl_mean_np = ctrl_mean
for g in use_genes:
    ho = ho_cells[g]
    Xt = np.asarray(X[ho][:, cols].todense(), dtype=np.float32)
    Xh = Xt[rng.choice(len(Xt), min(N_PICK, len(Xt)), replace=False)]
    d_tr = tr_delta[g]
    # 基线: ctrl + 基因训练均值 / 全局均值 / 原样
    base_cell = ctrl_mean_np
    cands = {
        "MAGWorld-cVAE": "model",
        "MeanDelta-shift": base_cell + d_tr,
        "GlobalMean-shift": base_cell + m_tr[keep_genes_mask],
        "Identity": base_cell,
    }
    for tag, p in cands.items():
        if tag == "MAGWorld-cVAE":
            with torch.no_grad():
                zc = torch.tensor(z_ctrl[:N_PICK])
                gi = torch.full((len(zc),), gidx_k[g], dtype=torch.long)
                dz = shift(zc, gi)
                xh = vae.dec(zc + dz).numpy()
            pred_mean = xh.mean(0)
            pc = np.mean([pearsonr(xh[i], Xh[i % len(Xh)])[0]
                          for i in range(len(xh))])
        else:
            pred_mean = p
            xh = np.tile(p, (N_PICK, 1))
            pc = np.mean([pearsonr(xh[i], Xh[i % len(Xh)])[0]
                          for i in range(len(xh))])
        true_delta = Xt.mean(0) - ctrl_mean_np
        pred_delta = pred_mean - ctrl_mean_np
        rows.append(dict(
            perturbation=g, model=tag, percell_pearson=pc,
            delta_pearson=safe_corr(pred_delta, true_delta),
            delta_mae_norm=np.abs(pred_delta - true_delta).mean()
            / (np.abs(true_delta).mean() + 1e-9),
            deg_top50_overlap=len(
                set(np.argsort(np.abs(true_delta))[-50:]) &
                set(np.argsort(np.abs(pred_delta))[-50:])) / 50))
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/percell_heldout.csv", index=False)
summ = res.groupby("model")[["percell_pearson", "delta_pearson",
                             "delta_mae_norm",
                             "deg_top50_overlap"]].mean()
summ.to_csv(f"{OUT}/percell_heldout_summary.csv")
print(summ, flush=True)

# ================= 4. 零样本未见基因路径 =================
print("\n=== zero-shot unseen genes (cVAE path) ===", flush=True)
P2 = np.load(PREP, allow_pickle=True)
dkeys = P2["delta_keys"].astype(str).tolist()
dmat = P2["delta_mat"]
delta_n = {k: dmat[i] for i, k in enumerate(dkeys)}
m2 = m_tr[keep_genes_mask]
zrows = []
test_usable = [g for g in test_genes
               if g in genes_k and g in delta_n]
pred_d_all, true_d_all = {}, {}
with torch.no_grad():
    zb = torch.tensor(z_ctrl[:3000])
    for g in test_usable:
        gi = torch.tensor([gidx_k[g]] * len(zb), dtype=torch.long)
        xh = vae.dec(zb + shift(zb, gi)).numpy()
        pred_delta = xh.mean(0) - ctrl_mean_np
        true_delta = delta_n[g][keep_genes_mask]
        pred_d_all[g] = pred_delta
        true_d_all[g] = true_delta
        zrows.append(dict(
            perturbation=g, model="MAGWorld-cVAE-zeroshot",
            delta_pearson=safe_corr(pred_delta, true_delta),
            delta_mae_norm=np.abs(pred_delta - true_delta).mean()
            / (np.abs(true_delta).mean() + 1e-9),
            deg_top50_overlap=len(
                set(np.argsort(np.abs(true_delta))[-50:]) &
                set(np.argsort(np.abs(pred_delta))[-50:])) / 50))
        zrows.append(dict(
            perturbation=g, model="MeanDelta",
            delta_pearson=safe_corr(m2, true_delta),
            delta_mae_norm=np.abs(m2 - true_delta).mean()
            / (np.abs(true_delta).mean() + 1e-9),
            deg_top50_overlap=len(
                set(np.argsort(np.abs(true_delta))[-50:]) &
                set(np.argsort(np.abs(m2))[-50:])) / 50))
zres = pd.DataFrame(zrows)
zres.to_csv(f"{OUT}/zeroshot_singles.csv", index=False)
print(zres.groupby("model")[["delta_pearson", "delta_mae_norm",
                             "deg_top50_overlap"]].mean(), flush=True)

torch.save({"vae": vae.state_dict(), "shift": shift.state_dict(),
            "genes": genes_k},
           f"{OUT}/cvae_model.pt")
print(f"\nV9 total {(time.time()-t0)/60:.1f} min")
print("MAGWORLD_V9_DONE")
