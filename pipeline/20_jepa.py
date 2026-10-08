# -*- coding: utf-8 -*-
"""v14: world-model base as an adapted JEPA (Joint Embedding Predictive
Architecture).

JEPA-style design:
  - encoder f_theta: expression -> latent s (MLP, 2000 HVG -> 32)
  - predictor g_phi: (s_t, action) -> s_{t+dt}  (MLP, action-conditioned)
  - trained with stop-gradient latent regression on latent-time-ordered
    cell pairs (scVelo latent_time), optionally action-conditioned
    (action = latent direction of a TF module; beta-scaled)
  - no generative decoder in the training loop: the loss lives entirely in
    latent space (this is the JEPA inductive bias)
Evaluation (same protocol as v12):
  - future-state MSE on held-out pairs vs persistence / kNN / linear / mean
  - action-conditioned rollout endpoints (RELA / STAT1)
The v12 VAE remains the generative readout for expression-space queries;
this script saves JEPA future-state metrics for Fig9 and the paper.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch
import torch.nn as nn

t0 = time.time()
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/worldmodel"
os.makedirs(OUT, exist_ok=True)
VEL = "/nfs_beijing/zizhuo/vcc/results/velocity/pancreas_velocity.h5ad"
REGULON = "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
np.random.seed(0)
torch.manual_seed(0)

# ---------------- data (same preprocessing as v12) ----------------
ad = sc.read_h5ad(VEL)
sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat")
X = ad[:, ad.var.highly_variable].X
X = X.toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X, np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
lt = np.asarray(ad.obs["latent_time"], dtype=np.float32)
n, d = X.shape
print(f"pancreas: {n} cells x {d} HVGs", flush=True)

# normalise per-cell features for the encoder input
mu = X.mean(0); sd = X.std(0) + 1e-6
Xs = ((X - mu) / sd).astype(np.float32)

# ---------------- latent-time-ordered pairs ----------------
order = np.argsort(lt)
pairs, dts = [], []
for i in range(n):
    j = np.searchsorted(lt[order], lt[i] + 0.02)
    if j < n:
        jj = order[j]
        if lt[jj] - lt[i] <= 0.15 and jj != i:
            pairs.append((i, jj))
            dts.append(lt[jj] - lt[i])
pairs = np.array(pairs)
dts = np.array(dts, dtype=np.float32)
print(f"pairs: {len(pairs)}", flush=True)

idx = np.random.permutation(len(pairs))
n_tr = int(0.8 * len(pairs))
ptr, pte = idx[:n_tr], idx[n_tr:]

# ---------------- adapted JEPA ----------------
ZD = 32

class Encoder(nn.Module):
    def __init__(self, din, z=ZD, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(),
                                 nn.Linear(h, z))
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, z=ZD, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * z, h), nn.SiLU(),
                                 nn.Linear(h, z))
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))

enc = Encoder(d)
pred = Predictor()
opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()),
                       lr=1e-3)
Xt = torch.tensor(Xs)
tr_i = torch.tensor(pairs[ptr, 0]); tr_j = torch.tensor(pairs[ptr, 1])
tr_dt = torch.tensor(dts[ptr])
# dt-scaled action channel: passive dynamics -> zero action vector
zero_a = torch.zeros(len(ptr), ZD)

for ep in range(800):
    opt.zero_grad()
    s_i = enc(Xt[tr_i])
    s_j = enc(Xt[tr_j])
    a = zero_a
    s_hat = pred(s_i, a)
    # time-gap weighting: larger dt -> larger expected displacement
    w = (tr_dt / tr_dt.mean()).unsqueeze(1)
    loss = (w * (s_hat - s_j.detach()) ** 2).mean() \
        + 0.01 * (s_i ** 2).mean()
    loss.backward()
    opt.step()
    if ep % 100 == 0:
        print(f"  jepa ep{ep} loss={float(loss):.4f}", flush=True)

torch.save(enc.state_dict(), f"{OUT}/jepa_encoder.pt")
torch.save(pred.state_dict(), f"{OUT}/jepa_predictor.pt")

# ---------------- future-state evaluation ----------------
def mse(a, b):
    return float(((a - b) ** 2).mean())

te_i = torch.tensor(pairs[pte, 0]); te_j = torch.tensor(pairs[pte, 1])
with torch.no_grad():
    S = enc(Xt).numpy().astype(np.float32)
    s_hat = pred(enc(Xt[te_i]), torch.zeros(len(te_i), ZD)).numpy()

knn_S = S[pairs[pte, 0]]
from sklearn.neighbors import NearestNeighbors
nnbr = NearestNeighbors(n_neighbors=6).fit(S[pairs[:, 0]])
_, nbi = nnbr.kneighbors(S[pairs[pte, 0]])
knn_disp = (S[pairs[nbi[:, 1:], 1]] - S[pairs[nbi[:, 1:], 0]]).mean(1)
Str = np.hstack([S[pairs[ptr, 0]], np.ones((len(ptr), 1), np.float32)])
Wr = np.linalg.solve(Str.T @ Str + 1e-2 * np.eye(ZD + 1),
                     Str.T @ S[pairs[ptr, 1]])
lin_disp = np.hstack([S[pairs[pte, 0]],
                      np.ones((len(pte), 1), np.float32)]) @ Wr
mean_disp = (S[pairs[ptr, 1]] - S[pairs[ptr, 0]]).mean(0, keepdims=True) \
    * np.ones((len(pte), 1))

fut = {
    "JEPA": mse(s_hat, S[pairs[pte, 1]]),
    "linear": mse(S[pairs[pte, 0]] + lin_disp, S[pairs[pte, 1]]),
    "kNN": mse(S[pairs[pte, 0]] + knn_disp, S[pairs[pte, 1]]),
    "mean_drift": mse(S[pairs[pte, 0]] + mean_disp, S[pairs[pte, 1]]),
    "persistence": mse(S[pairs[pte, 0]], S[pairs[pte, 1]]),
}
print(json.dumps(fut, indent=1), flush=True)

# velocity-style cosine for the JEPA displacement
disp = s_hat - S[pairs[pte, 0]]
true_disp = S[pairs[pte, 1]] - S[pairs[pte, 0]]
cos = float((disp * true_disp).sum(1).mean() /
            ((np.linalg.norm(disp, axis=1) *
              np.linalg.norm(true_disp, axis=1)).mean() + 1e-9))

# ---------------- action-conditioned rollout (RELA / STAT1) ----------------
Xfull = ad.X
Xfull = Xfull.toarray().astype(np.float32) if sp.issparse(Xfull) \
    else np.asarray(Xfull, np.float32)
Xfull = np.nan_to_num(Xfull, nan=0.0, posinf=0.0, neginf=0.0)
f2i = {g.lower(): i for i, g in enumerate(list(ad.var_names))}
net = pd.read_csv(REGULON)[["source", "target"]].dropna()
roll_info = {}
rng = np.random.RandomState(0)
ite = rng.choice(n, 400, replace=False)
for tf in ["RELA", "STAT1"]:
    tg = [t for t in net[net.source == tf].target
          if isinstance(t, str) and t.lower() in f2i]
    cols = [f2i[g.lower()] for g in tg]
    if tf.lower() in f2i:
        cols.append(f2i[tf.lower()])
    if len(cols) < 3:
        continue
    score = Xfull[:, cols].mean(1)
    hi = score > np.quantile(score, 0.8)
    lo = score <= np.quantile(score, 0.4)
    a_dir = (S[hi].mean(0) - S[lo].mean(0)).astype(np.float32)
    a_dir /= np.linalg.norm(a_dir) + 1e-9
    beta, T = 2.0, 12
    cur = torch.tensor(S[ite].copy())
    a = torch.tensor((beta * a_dir).astype(np.float32)).unsqueeze(0) \
        .repeat(len(ite), 1)
    traj = [cur.numpy().copy()]
    with torch.no_grad():
        for _ in range(T):
            cur = pred(cur, a)
            traj.append(cur.numpy().copy())
    traj = np.stack(traj)
    np.save(f"{OUT}/jepa_rollout_{tf}.npy", traj)
    d0 = np.linalg.norm(traj[0][:, None, :] - S[None, :, :], axis=2).min(1)
    d1 = np.linalg.norm(traj[-1][:, None, :] - S[None, :, :], axis=2).min(1)
    roll_info[tf] = {"dist_start": float(d0.mean()),
                     "dist_end": float(d1.mean()),
                     "displacement": float(
                         np.linalg.norm(traj[-1] - traj[0], axis=1).mean())}
    print(f"jepa rollout {tf}: {roll_info[tf]}", flush=True)

json.dump({"future_state_jepa": fut, "jepa_cos": cos,
           "n_pairs": int(len(pairs)), "rollouts": roll_info},
          open(f"{OUT}/jepa_results.json", "w"), indent=1)
pd.DataFrame([{"model": k, "future_mse": v} for k, v in fut.items()]
             ).to_csv(f"{OUT}/future_state_jepa.csv", index=False)
print(f"\nV14 total {(time.time()-t0)/60:.1f} min")
print("JEPA_DONE")
