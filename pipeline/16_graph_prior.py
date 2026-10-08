# -*- coding: utf-8 -*-
"""v11: GI-graph-prior operators on the Norman held-out benchmark.

Composite gene-interaction prior (external STRING/BioGRID unreachable from the
cluster; built from resources already staged):
  (i)   CollecTRI signed regulon edges (TF -> target, prior knowledge);
  (ii)  control-cell co-expression edges (top-k positive correlation per gene);
  (iii) perturbation epistasis edges from the Norman double-gene data
        (double conditions with strong synergy connect the two perturbed genes).

Models:
  A) GI-PROP   : SGC-style feature propagation mean_l(A^l E) + ridge
  B) GI-PPR    : personalised-PageRank diffusion profiles for train/test
                 genes (power iteration, sparse matvec), dual ridge
  C) GCN-direct: 2-layer GCN over the sparse graph, supervised on train-gene
                 deltas, direct prediction for test genes (GEARS-lite)
  D) GCN-emb   : penultimate GCN node embeddings + ridge operator
  E) GI-LAP    : one graph-propagation smoothing of training deltas + NW-corr
  F) MeanDelta / ridge-on-E references (from v7)

Metrics: delta_pearson, specific_pearson, discrimination rank (n=34).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.time()
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/graphprior"
os.makedirs(OUT, exist_ok=True)
PREP = "/nfs_beijing/zizhuo/vcc/results/magworld/norman/norman_prep.npz"
REGULON = "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"

P = np.load(PREP, allow_pickle=True)
genes = list(P["genes"]); g2i = {g: i for i, g in enumerate(genes)}
ng = len(genes)
E = P["E"]; G = P["G"]; m_tr = P["m_tr"]; D_tr = P["D_tr"]
train_genes = list(P["train_genes"]); test_genes = list(P["test_genes"])
delta_keys = list(P["delta_keys"]); delta_mat = P["delta_mat"]
d2i = {k: i for i, k in enumerate(delta_keys)}
D_te = np.vstack([delta_mat[d2i[g]] for g in test_genes])
tr_idx = np.array([g2i[g] for g in train_genes])
te_idx = np.array([g2i[g] for g in test_genes])
print(f"prep: {ng} genes, train={len(train_genes)}, test={len(test_genes)}",
      flush=True)

# ---------------- composite GI graph ----------------
rows, cols, vals = [], [], []

net = pd.read_csv(REGULON)[["source", "target", "weight"]].dropna()
for s, tg, w in net.itertuples(index=False):
    if s in g2i and tg in g2i and s != tg:
        i, j = g2i[s], g2i[tg]
        rows += [i, j]; cols += [j, i]; vals += [float(w), float(w)]
print(f"regulon directed entries {len(vals)}", flush=True)

K = 8
for i in range(ng):
    row = G[i].copy()
    row[i] = 0.0
    idx = np.argpartition(-row, K)[:K]
    for j in idx:
        if row[j] > 0.2:
            rows.append(i); cols.append(j); vals.append(float(row[j]))
print(f"+ coexpression, entries {len(vals)}", flush=True)

syn_edges = 0
for k in delta_keys:
    if "+" not in k:
        continue
    a, b = k.split("+")
    if a not in g2i or b not in g2i or a not in d2i or b not in d2i:
        continue
    obs = delta_mat[d2i[k]]
    add = delta_mat[d2i[a]] + delta_mat[d2i[b]]
    syn = np.linalg.norm(obs - add) / (np.linalg.norm(obs) + 1e-9)
    if syn > 0.4:
        i, j = g2i[a], g2i[b]
        rows += [i, j]; cols += [j, i]
        vals += [float(syn), float(syn)]
        syn_edges += 1
print(f"+ epistasis edges {syn_edges}", flush=True)

A = sp.coo_matrix((vals, (rows, cols)), shape=(ng, ng))
A = (A + A.T).tocsr()
A.setdiag(0); A.eliminate_zeros()
deg = np.asarray(np.abs(A).sum(1)).ravel()
dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
An = sp.diags(dinv) @ A @ sp.diags(dinv)          # signed normalised adjacency
An.data = np.clip(An.data, -1.0, 1.0)
print(f"graph: {ng} nodes, {An.nnz} edges, deg mean={deg.mean():.1f} "
      f"max={deg.max():.0f}  [{(time.time()-t0)/60:.1f} min]", flush=True)

# ---------------- metrics ----------------
def safe_corr(a, b):
    sa, sb = a.std(), b.std()
    return float(np.corrcoef(a, b)[0, 1]) if sa > 1e-12 and sb > 1e-12 else 0.0

def spec_corr(pred, true):
    return safe_corr(pred - m_tr, true - m_tr)

def disc_rank(pred_mat):
    n = pred_mat.shape[0]
    ranks = []
    for i in range(n):
        d = np.linalg.norm(pred_mat - D_te[i], axis=1)
        ranks.append(float((d < d[i]).sum() + 1))
    return float(np.mean(ranks))

results = []

def evaluate(pred_mat, name):
    rs = [safe_corr(pred_mat[i], D_te[i]) for i in range(len(test_genes))]
    sp_ = [spec_corr(pred_mat[i], D_te[i]) for i in range(len(test_genes))]
    rec = {"model": name,
           "delta_pearson": round(float(np.mean(rs)), 4),
           "delta_pearson_std": round(float(np.std(rs)), 4),
           "specific_pearson": round(float(np.mean(sp_)), 4),
           "discrimination": round(disc_rank(pred_mat), 2)}
    print(json.dumps(rec), flush=True)
    results.append(rec)

def dual_ridge_fit_predict(Xtr, Dtr, Xte, lam=10.0):
    n = Xtr.shape[0]
    Km = Xtr @ Xtr.T
    A_ = Km + lam * np.eye(n)
    alpha = np.linalg.solve(A_, Dtr)          # n x genes
    return Xte @ (Xtr.T @ alpha)

# ---------------- references ----------------
evaluate(np.tile(m_tr, (len(test_genes), 1)), "MeanDelta")
W_r = np.linalg.solve(E[tr_idx].T @ E[tr_idx] + 10 * np.eye(E.shape[1]),
                      E[tr_idx].T @ D_tr)
evaluate(E[te_idx] @ W_r, "ridge_on_E")

# ---------------- A) GI-PROP (SGC features) ----------------
Xs = [E]; H = E
for l in range(3):
    H = An @ H
    Xs.append(np.asarray(H))
X_prop = np.concatenate(Xs, axis=1)
evaluate(dual_ridge_fit_predict(X_prop[tr_idx], D_tr, X_prop[te_idx]),
         "GI_PROP_SGC")
print(f"prop done {(time.time()-t0)/60:.1f} min", flush=True)

# ---------------- B) GI-PPR diffusion profiles (train/test genes only) ----
alpha = 0.85
prof = np.zeros((len(tr_idx) + len(te_idx), ng), dtype=np.float32)
src = np.concatenate([tr_idx, te_idx])
for r, s in enumerate(src):
    e = np.zeros(ng, dtype=np.float32); e[s] = 1.0
    acc = e.copy(); cur = e.copy()
    for _ in range(12):
        cur = alpha * An.dot(cur)
        acc += cur
    prof[r] = (1 - alpha) * acc
X_ppr_tr = np.hstack([prof[:len(tr_idx)], E[tr_idx]])
X_ppr_te = np.hstack([prof[len(tr_idx):], E[te_idx]])
evaluate(dual_ridge_fit_predict(X_ppr_tr, D_tr, X_ppr_te, lam=100.0),
         "GI_PPR_diffusion")
print(f"ppr done {(time.time()-t0)/60:.1f} min", flush=True)

# ---------------- E) GI-LAP: graph-smoothed training deltas + NW ----------
D_sm = 0.5 * (D_tr + np.asarray(An[tr_idx][:, tr_idx] @ D_tr))
nw_pred = np.zeros((len(te_idx), D_tr.shape[1]))
for r, g in enumerate(test_genes):
    sim = G[g2i[g]][tr_idx]
    w = np.exp(sim / 0.1); w /= w.sum()
    nw_pred[r] = w @ D_sm
evaluate(nw_pred, "GI_LAP_smoothedNW")

# ---------------- C/D) GCN (sparse torch) ----------------
dev = "cuda" if torch.cuda.is_available() else "cpu"
An_coo = An.tocoo()
A_sp = torch.sparse_coo_tensor(
    torch.tensor(np.vstack([An_coo.row, An_coo.col]), dtype=torch.long),
    torch.tensor(An_coo.data, dtype=torch.float32),
    (ng, ng)).coalesce().to(dev)
X_in = torch.tensor(E[:, :-1], dtype=torch.float32, device=dev)
Dtr_t = torch.tensor(D_tr, dtype=torch.float32, device=dev)
tr_t = torch.tensor(tr_idx, dtype=torch.long, device=dev)
te_t = torch.tensor(te_idx, dtype=torch.long, device=dev)

class GCN(nn.Module):
    def __init__(self, din, hid=64, dout=64):
        super().__init__()
        self.w1 = nn.Linear(din, hid)
        self.w2 = nn.Linear(hid, dout)
    def forward(self, X, out_rows=None):
        H = torch.sparse.mm(A_sp, X)
        H = F.relu(self.w1(H))
        H = torch.sparse.mm(A_sp, H)
        if out_rows is not None:
            H = H[out_rows]
        return self.w2(H)

gcn = GCN(X_in.shape[1], 64, 64).to(dev)
opt = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-5)
out_head = nn.Linear(64, D_tr.shape[1]).to(dev)
opt2 = torch.optim.Adam(out_head.parameters(), lr=1e-3)
losses = []
for ep in range(300):
    gcn.train(); out_head.train()
    opt.zero_grad(); opt2.zero_grad()
    Htr = gcn(X_in, tr_t)
    pred = out_head(Htr)
    loss = F.mse_loss(pred, Dtr_t)
    loss.backward()
    opt.step(); opt2.step()
    losses.append(float(loss))
    if ep % 50 == 0:
        print(f"  gcn ep{ep} loss={float(loss):.4f}", flush=True)
gcn.eval(); out_head.eval()
with torch.no_grad():
    Hte = gcn(X_in, te_t)
    evaluate(out_head(Hte).cpu().numpy(), "GCN_direct")
    Htr64 = gcn(X_in, tr_t).cpu().numpy()
    Hte64 = Hte.cpu().numpy()
evaluate(dual_ridge_fit_predict(Htr64, D_tr, Hte64, lam=10.0), "GCN_emb_ridge")
print(f"gcn done {(time.time()-t0)/60:.1f} min", flush=True)

# ---------------- save ----------------
df = pd.DataFrame(results)
df.to_csv(f"{OUT}/graphprior_norman.csv", index=False)
json.dump({"graph": {"nodes": int(ng), "edges": int(An.nnz),
                     "epistasis_edges": int(syn_edges),
                     "deg_mean": float(deg.mean())},
           "gcn_train_loss_first": losses[0], "gcn_train_loss_last": losses[-1]},
          open(f"{OUT}/graphprior_config.json", "w"), indent=1)
print(df.to_string(), flush=True)
print(f"\nV11 total {(time.time()-t0)/60:.1f} min")
print("GRAPHPRIOR_DONE")
