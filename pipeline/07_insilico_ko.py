# -*- coding: utf-8 -*-
"""MAGWorld 实验三: in silico TF KO + 功能重评分 (dat17 CRISPRi)
- World VAE 在 dat17 对照细胞上预训练
- 对目标 TF 基因 t: 沿解码器梯度方向 dz_t = mean_z ∇_z decode(z)_t 做潜空间干预,
  强度标定到目标基因表达下降 50% (in silico knockdown)
- 对预测状态用 CollecTRI + ULM 重评分 TF 活性, 检验:
  (i) KO t 后 t 自身活性下降; (ii) 下游 TF 模块的连锁变化
输出: results/magworld/insilico_ko_tf_activity.csv, insilico_ko_summary.csv
"""
import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import decoupler as dc

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

CORE = "/nfs_beijing/zizhuo/vcc/results/core"
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld"
RES = "/nfs_beijing/zizhuo/vcc/data/resources"


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


print("加载 dat17 (对照细胞) ...", flush=True)
ad = sc.read_h5ad(f"{CORE}/dat17_processed.h5ad")
tgt_all = ad.obs["target"].astype(str)
is_ctrl = (tgt_all.isna() | (tgt_all.str.lower() == "nan") |
           (ad.obs["perturbation"].astype(str).str.lower() == "control"))
ad = ad[is_ctrl].copy()
X = ad.raw.X.toarray() if hasattr(ad.raw.X, "toarray") else np.asarray(ad.raw.X)
X = pd.DataFrame(X, index=ad.obs_names, columns=ad.raw.var_names)
# 用全基因空间 (与 03 一致, 保证 CollecTRI 各 TF 有足够靶点)
genes = [g for g in X.columns if X[g].sum() != 0]
CAND = ["RELA", "NFKB1", "JUN", "FOS", "EGR1", "GATA3", "IRF1", "STAT1",
        "JUNB", "NFATC1"]
genes += [t for t in CAND if t in X.columns and t not in genes]
Xv = X[genes].values
g2i = {g: i for i, g in enumerate(genes)}
print(f"dat17: {Xv.shape[0]} cells x {len(genes)} genes", flush=True)

Xt = torch.tensor(Xv, dtype=torch.float32)
model = VAE(len(genes))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
n = Xt.shape[0]
for ep in range(30):
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, 1024):
        xb = Xt[perm[i:i + 1024]]
        xr, mu, lv = model(xb)
        rec = nn.functional.mse_loss(xr, xb)
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        loss = rec + 1e-3 * kl
        opt.zero_grad(); loss.backward(); opt.step()
        tot += rec.item() * len(xb)
    if (ep + 1) % 10 == 0:
        print(f"  [vae-pbmc] {ep+1}/30 recon={tot/n:.4f}", flush=True)
model.eval()

with torch.no_grad():
    z_all = model.encode(Xt)
    center = z_all.mean(0, keepdim=True)
Z = (z_all - center).numpy()
Zt = torch.tensor(Z, dtype=torch.float32, requires_grad=False)

net = pd.read_csv(f"{RES}/CollecTRI_regulons.csv")[["source", "target",
                                                    "weight"]].dropna()
ko_genes = [t for t in CAND if t in g2i]
print("KO genes:", ko_genes, flush=True)

# 基线 (干预前) TF 活性
ad0 = sc.AnnData(Xv)
ad0.var_names = genes
ad0.obs = ad.obs.copy()
dc.mt.ulm(ad0, net, verbose=False)
_sc = ad0.obsm["score_ulm"]
tf_names = (list(_sc.columns) if hasattr(_sc, "columns")
            else [f"TF{i}" for i in range(np.asarray(_sc).shape[1])])
base = pd.DataFrame(np.asarray(_sc), columns=tf_names,
                    index=ad0.obs_names)
print(f"baseline ULM: {base.shape}, NaN={int(base.isna().sum().sum())}",
      flush=True)

records, summary = [], []
for t in ko_genes:
    ti = g2i[t]
    # 潜空间干预方向: 解码器对目标基因表达的梯度 (取 batch 均值)
    dz = torch.zeros(32)
    B = 512
    for i in range(0, len(Zt), B):
        zb = Zt[i:i + B].clone().requires_grad_(True)
        expr_t = model.decode(zb)[:, ti].sum()
        expr_t.backward()
        dz += zb.grad.sum(0).detach()
        zb.grad.zero_()
    dz = dz / len(Zt)
    dz = dz / (dz.norm() + 1e-9)
    # 标定强度: 目标基因平均表达下降 ~50%
    with torch.no_grad():
        base_expr = model.decode(Zt[:1024])[:, ti].mean().item()
    lo, hi = 0.0, 30.0
    for _ in range(25):
        mid = (lo + hi) / 2
        with torch.no_grad():
            e = model.decode(Zt[:1024] - mid * dz)[:, ti].mean().item()
        if e < 0.5 * base_expr:
            hi = mid
        else:
            lo = mid
    s = hi
    with torch.no_grad():
        X_ko = model.decode(Zt - s * dz).numpy()
    print(f"  X_ko NaN={int(np.isnan(X_ko).any())} "
          f"range=({np.nanmin(X_ko):.2f},{np.nanmax(X_ko):.2f})", flush=True)
    adk = sc.AnnData(X_ko)
    adk.var_names = genes
    dc.mt.ulm(adk, net, verbose=False)
    _sk = np.asarray(adk.obsm["score_ulm"])
    print(f"  KO ULM raw shape={_sk.shape} "
          f"NaN={int(np.isnan(_sk).any())}", flush=True)
    ko_act = pd.DataFrame(_sk[:, :base.shape[1]], columns=tf_names,
                          index=ad0.obs_names)
    delta = (ko_act - base).mean(0)
    for tf in delta.index:
        records.append(dict(ko_target=t, tf=tf,
                            base_activity=base[tf].mean(),
                            ko_activity=ko_act[tf].mean(),
                            delta_activity=delta[tf]))
    self_d = delta[t] if t in delta.index else np.nan
    rank = int((delta < self_d).sum()) if not np.isnan(self_d) else -1
    summary.append(dict(ko_target=t, scale=s,
                        target_gene_expr_drop=0.5,
                        self_tf_delta=self_d,
                        self_tf_downstream_rank=rank,
                        n_downstream_tf_decreased=int((delta < -0.05).sum())))
    print(f"[KO {t}] scale={s:.2f} self_delta={self_d:.3f} "
          f"rank={rank}", flush=True)

pd.DataFrame(records).to_csv(f"{OUT}/insilico_ko_tf_activity.csv",
                             index=False)
pd.DataFrame(summary).to_csv(f"{OUT}/insilico_ko_summary.csv", index=False)
print("KO_DONE")
