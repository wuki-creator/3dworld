# -*- coding: utf-8 -*-
"""v13: GEARS (official, cell-gears 0.1.2) as SOTA comparison, Norman tier.

Protocol alignment with the MAGWorld benchmark:
  - same gene space as norman_prep.npz (3,000 HVG + all perturbed genes)
  - same seed-0 split: train = 68 training singles, test = 34 held-out singles
    (GEARS custom split; ~10 train genes reserved as val for early stopping)
  - same metrics computed against the same observed deltas (norman_prep)

Environment adaptations (documented in paper):
  - curated GO gene-set download (Harvard Dataverse) unreachable from the
    cluster -> gene2go built from signed CollecTRI regulons (gene -> set of
    regulating TFs as pseudo-GO terms); GEARS GO-graph therefore encodes
    shared-regulator structure
  - default_pert_graph=False (perturbation graph restricted to dataset genes)
  - PyTorch is CPU-only on this node -> training cells subsampled to ~30k
    stratified by condition (ctrl kept in full)
"""
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

warnings.filterwarnings("ignore")
np.random.seed(0)
t0 = time.time()

H5AD = "/nfs_beijing/zizhuo/vcc/data/norman/NormanWeissman2019_filtered.h5ad"
PREP = "/nfs_beijing/zizhuo/vcc/results/magworld/norman/norman_prep.npz"
REGULON = "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/gears"
PDIR = os.path.join(OUT, "pertdata")
os.makedirs(PDIR, exist_ok=True)
MAX_CELLS = 30000

P = np.load(PREP, allow_pickle=True)
genes = list(P["genes"])
train_genes = list(P["train_genes"])
test_genes = list(P["test_genes"])
delta_keys = list(P["delta_keys"])
delta_mat = P["delta_mat"]
m_tr = P["m_tr"]
d2i = {k: i for i, k in enumerate(delta_keys)}
D_te = np.vstack([delta_mat[d2i[g]] for g in test_genes])

# ---------- pseudo-GO gene sets from CollecTRI ----------
g2go_pkl = os.path.join(PDIR, "gene2go_all.pkl")
if not os.path.exists(g2go_pkl):
    net = pd.read_csv(REGULON)[["source", "target"]].dropna()
    g2go = {}
    for s, tg in net.itertuples(index=False):
        if isinstance(tg, str):
            g2go.setdefault(tg, set()).add("TF:" + str(s))
    for g in genes:                       # ensure every gene has a term
        g2go.setdefault(g, set()).add("SELF:" + g)
    pickle.dump(g2go, open(g2go_pkl, "wb"))
    print(f"gene2go built: {len(g2go)} genes", flush=True)

# ---------- build GEARS-format adata, subsampled ----------
print("加载 Norman h5ad ...", flush=True)
ad = sc.read_h5ad(H5AD)
X = ad.X.tocsr() if sparse.isspmatrix_csc(ad.X) else ad.X
labels = ad.obs["perturbation"].astype(str).values
nperts = ad.obs["nperts"].astype(int).values
r2i = {g: i for i, g in enumerate(ad.var_names.astype(str))}
cols = np.array([r2i[g] for g in genes])

def to_cond(lab, npt):
    lab = str(lab).strip()
    if npt == 0 or lab == "control":
        return "ctrl"
    if npt == 2 and "_" in lab:
        a, b = sorted(lab.split("_"))
        return f"{a}+{b}"
    return lab

cond = np.array([to_cond(l, n) for l, n in zip(labels, nperts)])
rng = np.random.RandomState(0)
keep_idx = []
ctrl_idx = np.where(cond == "ctrl")[0]
keep_idx += list(ctrl_idx)
budget = MAX_CELLS - len(ctrl_idx)
perts = [c for c in pd.unique(cond) if c != "ctrl"]
cap = max(20, budget // max(1, len(perts)))
for c in perts:
    sel = np.where(cond == c)[0]
    if len(sel) > cap:
        sel = rng.choice(sel, cap, replace=False)
    keep_idx += list(sel)
keep_idx = np.array(sorted(keep_idx))
print(f"subsampled cells: {len(keep_idx)} (ctrl={len(ctrl_idx)}, "
      f"cap/cond={cap})", flush=True)

subX = X[keep_idx][:, cols].astype(np.float32)
if not sparse.issparse(subX):
    subX = sparse.csr_matrix(subX)
sub = sc.AnnData(subX)
sub.obs["condition"] = pd.Categorical(cond[keep_idx])
sub.obs["cell_type"] = "K562"
sub.var_names = genes
sub.var["gene_name"] = genes
sub.uns["log1p"] = {"base": None}

from gears import PertData, GEARS

pert_data = PertData(PDIR, default_pert_graph=False)
print("PertData ready; default_pert_graph=False", flush=True)
pert_data.new_data_process(dataset_name="gearsnorman", adata=sub,
                           skip_calc_de=False)
print(f"data processed {(time.time()-t0)/60:.1f} min", flush=True)

# ---------- custom split: same seed-0 MAGWorld partition ----------
val_genes = sorted(rng.choice(train_genes, 10, replace=False))
tr_genes = sorted(set(train_genes) - set(val_genes))
train_conds = ["ctrl"] + tr_genes
val_conds = val_genes
test_conds = list(test_genes)
split_dict = {"train": train_conds, "val": val_conds, "test": test_conds}
split_path = os.path.join(OUT, "magworld_split.pkl")
pickle.dump(split_dict, open(split_path, "wb"))
pert_data.prepare_split(split="custom", split_dict_path=split_path)
pert_data.get_dataloader(batch_size=64, test_batch_size=512)
print(f"split: train={len(train_conds)} val={len(val_conds)} "
      f"test={len(test_conds)}", flush=True)

gears_model = GEARS(pert_data, device="cpu")
gears_model.model_initialize()
gears_model.train(epochs=20)
print(f"training done {(time.time()-t0)/60:.1f} min", flush=True)
gears_model.save_model(f"{OUT}/model")   # 先存档，防止后续步骤失败重训

# ---------- predict held-out singles, compute MAGWorld metrics ----------
avail = [g for g in test_genes if g in gears_model.pert_list]
print(f"predictable test genes: {len(avail)}/{len(test_genes)}", flush=True)
res = gears_model.predict([[g] for g in avail])
results_pred = res[0] if isinstance(res, (tuple, list)) else res
print("predict returned:", type(res), flush=True)

_ctrlX = sub.X[sub.obs.condition.values == "ctrl"]
ctrl_mean = np.asarray(_ctrlX.toarray() if sparse.issparse(_ctrlX)
                       else _ctrlX).mean(0)

def safe_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

rows = []
pred_mat = []
for g in avail:
    pred_delta = np.asarray(results_pred[g], dtype=np.float64).ravel() \
        - ctrl_mean
    true_delta = D_te[test_genes.index(g)]
    pred_mat.append(pred_delta)
    rows.append({
        "gene": g,
        "delta_pearson": safe_corr(pred_delta, true_delta),
        "specific_pearson": safe_corr(pred_delta - m_tr, true_delta - m_tr),
    })
pred_mat = np.vstack(pred_mat)
disc = []
for i in range(len(avail)):
    d = np.linalg.norm(pred_mat - D_te[test_genes.index(avail[i])], axis=1)
    disc.append(float((d < d[i]).sum() + 1))
for r, dk in zip(rows, disc):
    r["disc_rank"] = dk
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/gears_metrics_test34.csv", index=False)
summary = {
    "model": "GEARS_official",
    "n_test": len(avail),
    "delta_pearson": float(df.delta_pearson.mean()),
    "delta_pearson_std": float(df.delta_pearson.std()),
    "specific_pearson": float(df.specific_pearson.mean()),
    "discrimination": float(df.disc_rank.mean()),
    "train_conditions": len(train_conds),
    "train_cells": int(len(keep_idx)),
    "epochs": 20,
    "device": "cpu",
}
print(json.dumps(summary, indent=1), flush=True)
json.dump(summary, open(f"{OUT}/gears_summary.json", "w"), indent=1)
print(f"\nV13 total {(time.time()-t0)/60:.1f} min")
print("GEARS_DONE")
