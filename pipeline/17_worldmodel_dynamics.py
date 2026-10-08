# -*- coding: utf-8 -*-
"""v12: world-model底座 = VAE latent state + learned latent dynamics (flow).

Pipeline (all on cluster):
  1. Train a VAE (HVG space) on the pancreas endocrinogenesis dataset; the
     32-d latent z is the world state (the 'world model底座').
  2. Estimate per-cell latent velocity v_z by local-linearisation of the
     encoder around each cell (dual ridge on kNN expression differences).
  3. Learn the transition field f_theta: z -> dz/dt (MLP) on held-in cells.
  4. Evaluate on held-out cells:
       (a) velocity MSE / cosine vs baselines (persistence, mean-drift,
           kNN regressor, linear ridge);
       (b) future-state prediction over latent-time-ordered pairs:
           z_j ~ z_i + f(z_i) * dt  vs baselines.
  5. Intervention rollout: an in silico TF action perturbs the field,
       z_{t+1} = z_t + (f(z_t) + B a) dt; show RELA-like action diverts the
       trajectory to a distinct region of the manifold (saved figure).
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
import torch.nn.functional as F

t0 = time.time()
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/worldmodel"
os.makedirs(OUT, exist_ok=True)
VEL = "/nfs_beijing/zizhuo/vcc/results/velocity/pancreas_velocity.h5ad"
np.random.seed(0); torch.manual_seed(0)

# ---------------- data ----------------
ad = sc.read_h5ad(VEL)
# 01 号管线产物已是 lognorm，直接复用，避免二次 log 变换
sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat")
X = ad[:, ad.var.highly_variable].X
if sp.issparse(X):
    X = X.toarray().astype(np.float32)
else:
    X = np.asarray(X, dtype=np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
lt = np.asarray(ad.obs["latent_time"], dtype=np.float32)
Vx = np.asarray(ad[:, ad.var.highly_variable].layers["velocity"])
if sp.issparse(Vx):
    Vx = Vx.toarray()
Vx = np.nan_to_num(np.asarray(Vx, dtype=np.float32))
n, d = X.shape
print(f"pancreas: {n} cells x {d} HVGs, latent_time range "
      f"[{lt.min():.2f},{lt.max():.2f}]", flush=True)

# ---------------- VAE world底座 ----------------
H = 128; ZD = 32
X_t = torch.tensor(X)
class VAE(nn.Module):
    def __init__(self, din, h=H, z=ZD):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(din, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, z); self.lv = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, din))
    def forward(self, x):
        h = self.enc(x)
        mu, logv = self.mu(h), self.lv(h).clamp(-8, 4)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
        return self.dec(z), mu, logv

vae = VAE(d)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
for ep in range(200):
    opt.zero_grad()
    xr, mu, logv = vae(X_t)
    rec = F.mse_loss(xr, X_t)
    kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(1).mean() / d
    loss = rec + 0.1 * kl
    loss.backward(); opt.step()
    if ep % 50 == 0:
        print(f"  vae ep{ep} rec={float(rec):.4f}", flush=True)
vae.eval()
with torch.no_grad():
    Z = vae.enc(X_t); Z = vae.mu(Z).numpy().astype(np.float32)
np.save(f"{OUT}/vae_latent.npy", Z)
torch.save(vae.state_dict(), f"{OUT}/worldmodel_vae.pt")
print(f"vae done {(time.time()-t0)/60:.1f} min", flush=True)

# ---------------- local-linear latent velocity ----------------
from sklearn.neighbors import NearestNeighbors
k = 20
nn_fit = NearestNeighbors(n_neighbors=k + 1).fit(X)
_, nbr = nn_fit.kneighbors(X)
lam = 1.0
Vz = np.zeros((n, ZD), dtype=np.float32)
for i in range(n):
    nb = nbr[i, 1:]
    dX = X[nb] - X[i]
    dZ = Z[nb] - Z[i]
    Km = dX @ dX.T + lam * np.eye(k)
    alpha = np.linalg.solve(Km, dX @ Vx[i])
    Vz[i] = dZ.T @ alpha
spd = np.linalg.norm(Vz, axis=1)
print(f"latent velocity: mean|v_z|={spd.mean():.3f} "
      f"median={np.median(spd):.3f}", flush=True)

# ---------------- split ----------------
idx = np.random.permutation(n)
n_tr = int(0.8 * n)
itr, ite = idx[:n_tr], idx[n_tr:]

# ---------------- flow model ----------------
class Flow(nn.Module):
    def __init__(self, z=ZD, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, z))
    def forward(self, z):
        return self.net(z)

flow = Flow()
opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
Ztr = torch.tensor(Z[itr]); Vtr = torch.tensor(Vz[itr])
for ep in range(600):
    opt.zero_grad()
    loss = F.mse_loss(flow(Ztr), Vtr)
    loss.backward(); opt.step()
    if ep % 100 == 0:
        print(f"  flow ep{ep} loss={float(loss):.5f}", flush=True)
torch.save(flow.state_dict(), f"{OUT}/worldmodel_flow.pt")

def cos(a, b):
    na = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float((a * b).sum(1).mean() / 1.0 / (np.abs(na).mean()))

# baselines
mean_v = Vz[itr].mean(0, keepdims=True) * np.ones((len(ite), 1))
knn_v = Vz[nbr[ite, 1:6]].mean(1)                       # kNN velocity
Ztr_r = np.hstack([Z[itr], np.ones((len(itr), 1))])
Wr = np.linalg.solve(Ztr_r.T @ Ztr_r + 1e-2 * np.eye(ZD + 1),
                     Ztr_r.T @ Vz[itr])
lin_v = np.hstack([Z[ite], np.ones((len(ite), 1))]) @ Wr
with torch.no_grad():
    mlp_v = flow(torch.tensor(Z[ite])).numpy()

def mse(a, b):
    return float(((a - b) ** 2).mean())

vel_res = {
    "flow_mlp":  {"mse": mse(mlp_v, Vz[ite]),
                 "cos": float((mlp_v * Vz[ite]).sum(1).mean() /
                              (np.linalg.norm(mlp_v, axis=1) *
                               np.linalg.norm(Vz[ite], axis=1) + 1e-9).mean())},
    "linear_ridge": {"mse": mse(lin_v, Vz[ite]),
                     "cos": float((lin_v * Vz[ite]).sum(1).mean() /
                                  (np.linalg.norm(lin_v, axis=1) *
                                   np.linalg.norm(Vz[ite], axis=1) + 1e-9).mean())},
    "kNN_velocity": {"mse": mse(knn_v, Vz[ite]),
                     "cos": float((knn_v * Vz[ite]).sum(1).mean() /
                                  (np.linalg.norm(knn_v, axis=1) *
                                   np.linalg.norm(Vz[ite], axis=1) + 1e-9).mean())},
    "mean_drift": {"mse": mse(mean_v, Vz[ite]), "cos": 0.0},
    "persistence_v0": {"mse": mse(np.zeros_like(Vz[ite]), Vz[ite]), "cos": 0.0},
}
print(json.dumps(vel_res, indent=1), flush=True)

# ---------------- future-state pairs ----------------
order = np.argsort(lt)
# for each cell, nearest cell in latent time with dt in [0.02, 0.15]
pairs = []
for i in range(n):
    j = np.searchsorted(lt[order], lt[i] + 0.02)
    jj = order[j] if j < n else -1
    if jj >= 0 and lt[jj] - lt[i] <= 0.15 and jj != i:
        pairs.append((i, jj, lt[jj] - lt[i]))
pairs = np.array(pairs, dtype=int if False else object)
pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
pdt = np.array([p[2] for p in pairs], dtype=np.float32)
print(f"future pairs: {len(pi)}", flush=True)
# calibrate time scale: s = median ||dz|| / ||v_z||
dz = Z[pj] - Z[pi]
s = np.median(np.linalg.norm(dz, axis=1) /
              (np.linalg.norm(Vz[pi], axis=1) + 1e-9))
print(f"time-scale calibration s={s:.3f}", flush=True)
dt_eff = pdt * s

ite_set = set(ite.tolist())
te_mask = np.array([i in ite_set for i in pi])
with torch.no_grad():
    f_pi = flow(torch.tensor(Z[pi])).numpy() * dt_eff[:, None]
knn_full = Vz[nbr[pi, 1:6]].mean(1)
lin_full = np.hstack([Z[pi], np.ones((len(pi), 1))]) @ Wr
fut = {
    "flow": mse(Z[pi] + f_pi, Z[pj]),
    "persistence": mse(Z[pi], Z[pj]),
    "mean_drift": mse(Z[pi] + mean_v[0] * dt_eff[:, None], Z[pj]),
    "kNN": mse(Z[pi] + knn_full * dt_eff[:, None], Z[pj]),
    "linear": mse(Z[pi] + lin_full * dt_eff[:, None], Z[pj]),
}
print(json.dumps(fut, indent=1), flush=True)

# ---------------- intervention rollout ----------------
# action = latent direction that separates high vs low TF-module expression,
# scored on the FULL gene space (not only HVGs)
Xfull = ad.X
if sp.issparse(Xfull):
    Xfull = Xfull.toarray().astype(np.float32)
else:
    Xfull = np.asarray(Xfull, dtype=np.float32)
Xfull = np.nan_to_num(Xfull, nan=0.0, posinf=0.0, neginf=0.0)
full_names = list(ad.var_names)
f2i = {g.lower(): i for i, g in enumerate(full_names)}
net = pd.read_csv(
    "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
)[["source", "target", "weight"]].dropna()
for tf in ["RELA", "NFKB1", "STAT1"]:
    tg = [t for t in net[net.source == tf].target
          if isinstance(t, str) and t.lower() in f2i]
    cols = [f2i[g.lower()] for g in tg]
    if tf.lower() in f2i:
        cols.append(f2i[tf.lower()])
    if len(cols) < 3:
        print(f"rollout {tf}: only {len(cols)} module genes, skipped",
              flush=True)
        continue
    score = Xfull[:, cols].mean(1)
    hi = score > np.quantile(score, 0.8)
    lo = score <= np.quantile(score, 0.4)
    a_dir = (Z[hi].mean(0) - Z[lo].mean(0)).astype(np.float32)
    a_dir /= np.linalg.norm(a_dir) + 1e-9
    # rollout from held-out cells: T steps, dt small; action strength beta
    T = 12; beta = 2.0
    z0 = Z[ite[:400]].copy()
    zr = [z0.copy()]
    with torch.no_grad():
        cur = torch.tensor(z0)
        for _ in range(T):
            v = flow(cur).numpy() * 0.1 + beta * a_dir * 0.1
            cur = torch.tensor(cur.numpy() + v)
            zr.append(cur.numpy().copy())
    zr = np.stack(zr)                      # (T+1, 400, ZD)
    np.save(f"{OUT}/rollout_{tf}.npy", zr)
    # on-manifold check: decode endpoints, reconstruction error
    with torch.no_grad():
        rec0 = vae.dec(torch.tensor(zr[0])).numpy()
        rec1 = vae.dec(torch.tensor(zr[-1])).numpy()
    # distance to real cells (manifold membership)
    d0 = np.linalg.norm(zr[0][:, None, :] - Z[None, :, :], axis=2).min(1)
    d1 = np.linalg.norm(zr[-1][:, None, :] - Z[None, :, :], axis=2).min(1)
    print(f"rollout {tf}: dist-to-manifold start={d0.mean():.2f} "
          f"end={d1.mean():.2f}; displacement="
          f"{np.linalg.norm(zr[-1] - zr[0], axis=1).mean():.2f}", flush=True)

# ---------------- save ----------------
json.dump({"velocity_pred": vel_res, "future_state": fut,
           "time_scale": float(s), "n_future_pairs": int(len(pi)),
           "vae_recon": float(rec)},
          open(f"{OUT}/worldmodel_results.json", "w"), indent=1)
pd.DataFrame([{"model": k, **v} for k, v in vel_res.items()]
             ).to_csv(f"{OUT}/velocity_prediction.csv", index=False)
pd.DataFrame([{"model": k, "future_mse": v} for k, v in fut.items()]
             ).to_csv(f"{OUT}/future_state.csv", index=False)
print(f"\nV12 total {(time.time()-t0)/60:.1f} min")
print("WORLDMODEL_DONE")
