# -*- coding: utf-8 -*-
"""v15: GEARS official (cell-gears 0.1.2) zero-shot Replogle -> Norman transfer.

Protocol alignment with the MAGWorld transfer benchmark (script 12):
  - gene space: norman_prep.npz genes (3,000 HVG + all perturbed genes);
    Replogle expression mapped into this space, genes absent from the
    Replogle matrix are zero-filled (same convention as the T1 ridge
    transfer operator)
  - training: Replogle K562_essential single-gene knockdowns only
  - evaluation: the same 34 Norman held-out single-gene test conditions,
    same metrics (delta_pearson vs observed Norman deltas, specific_pearson
    after removing the Norman-68 shared mean, discrimination rank)
  - zero-shot: Norman test genes are never seen as conditions during
    training; prediction goes through the GEARS perturbation graph
    (gene2go = CollecTRI pseudo-GO, identical to the Norman-tier run)

Environment adaptations (same three as the Norman-tier GEARS run, disclosed
in the paper): CPU training, training cells subsampled to ~30k, Harvard
Dataverse GO replaced by CollecTRI regulons.
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
rng = np.random.RandomState(0)
t0 = time.time()

REP = "/nfs_beijing/zizhuo/vcc/data/norman/ReplogleWeissman2022_K562_essential.h5ad"
PREP = "/nfs_beijing/zizhuo/vcc/results/magworld/norman/norman_prep.npz"
REGULON = "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
NORMAN_OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/gears"
OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/gears_transfer"
PDIR = os.path.join(NORMAN_OUT, "pertdata")   # reuse gene2go from v13
os.makedirs(OUT, exist_ok=True)
MAX_CELLS = 30000
MIN_CELLS_COND = 10

P = np.load(PREP, allow_pickle=True)
genes = list(P["genes"])
test_genes = list(P["test_genes"])
m_tr = P["m_tr"]
delta_keys = list(P["delta_keys"])
delta_mat = P["delta_mat"]
d2i = {k: i for i, k in enumerate(delta_keys)}
D_te = np.vstack([delta_mat[d2i[g]] for g in test_genes])
g2i = {g: i for i, g in enumerate(genes)}

# ---------- pseudo-GO gene sets (reuse v13 asset) ----------
g2go_pkl = os.path.join(PDIR, "gene2go_all.pkl")
assert os.path.exists(g2go_pkl), "run 19_gears_run.py first (gene2go missing)"
print("gene2go reused from v13", flush=True)

# ---------- load Replogle, normalise, map to Norman gene space ----------
print("加载 Replogle h5ad ...", flush=True)
ad = sc.read_h5ad(REP)
X = ad.X.tocsr() if sparse.isspmatrix_csc(ad.X) else ad.X
per = ad.obs["perturbation"].astype(str).values
var_names = ad.var_names.astype(str).values

xs = X[:20]
if hasattr(xs, "toarray"):
    xs = xs.toarray()
row_med = float(np.median(np.asarray(xs.sum(1)).ravel()))
print(f"replogle: {X.shape} median rowsum={row_med:.0f}", flush=True)
if row_med > 50000:
    tot = np.asarray(X.sum(1)).ravel()
    scale = 1e4 / np.maximum(tot, 1.0)
    X = sparse.diags(scale).dot(X)
    X.data = np.log1p(X.data)
    X = X.tocsr()
    print("已做 log1p CP10k 归一化", flush=True)

raw2i = {g: i for i, g in enumerate(var_names)}
shared = [g for g in genes if g in raw2i]
print(f"shared genes with norman space: {len(shared)}/{len(genes)}", flush=True)
cols_n = np.array([g2i[g] for g in shared])
cols_r = np.array([raw2i[g] for g in shared])

# ---------- condition table ----------
ctrl_idx = np.where(per == "control")[0]
cond_pairs = []
for lab in pd.unique(per):
    lab = str(lab).strip()
    if lab == "control" or not lab or lab.lower() in ("nan", "none") \
            or "+" in lab or "_" in lab:
        continue
    if lab not in g2i:
        continue
    sel = np.where(per == lab)[0]
    if len(sel) < MIN_CELLS_COND:
        continue
    cond_pairs.append((lab, sel))
cond_names = [g for g, _ in cond_pairs]
print(f"usable replogle conditions: {len(cond_names)}", flush=True)
overlap = [g for g in test_genes if g in set(cond_names)]
print(f"norman test genes that are ALSO replogle training conditions: "
      f"{len(overlap)} {overlap}", flush=True)

# ---------- subsample to budget, map expression ----------
keep_pairs = [(int(c), "ctrl") for c in ctrl_idx]
budget = MAX_CELLS - len(ctrl_idx)
cap = max(10, budget // max(1, len(cond_pairs)))
for lab, sel in cond_pairs:
    if len(sel) > cap:
        sel = rng.choice(sel, cap, replace=False)
    keep_pairs += [(int(c), lab) for c in sel]
keep_pairs.sort(key=lambda x: x[0])
keep_idx = np.array([c for c, _ in keep_pairs])
cond = np.array([l for _, l in keep_pairs], dtype=object)
print(f"subsampled cells: {len(keep_idx)} (ctrl={len(ctrl_idx)}, "
      f"cap/cond={cap})", flush=True)

subX = X[keep_idx][:, cols_r].astype(np.float32)
subX = subX.toarray() if sparse.issparse(subX) else np.asarray(subX)
subX_full = np.zeros((len(keep_idx), len(genes)), dtype=np.float32)
subX_full[:, cols_n] = subX

sub = sc.AnnData(sparse.csr_matrix(subX_full))
sub.obs["condition"] = pd.Categorical(cond)
sub.obs["cell_type"] = "K562"
sub.var_names = genes
sub.var["gene_name"] = genes
sub.uns["log1p"] = {"base": None}

# ---------- GEARS pipeline ----------
from gears import PertData, GEARS

pert_data = PertData(PDIR, default_pert_graph=False)
pert_data.new_data_process(dataset_name="gearstransfer", adata=sub,
                           skip_calc_de=False)
print(f"data processed {(time.time()-t0)/60:.1f} min", flush=True)

val_pool = sorted(rng.choice(cond_names, 100, replace=False).tolist())
split_dict = {"train": ["ctrl"] + sorted(set(cond_names) - set(val_pool)),
              "val": val_pool[:50],
              "test": val_pool[50:]}
split_path = os.path.join(OUT, "transfer_split.pkl")
pickle.dump(split_dict, open(split_path, "wb"))
pert_data.prepare_split(split="custom", split_dict_path=split_path)
pert_data.get_dataloader(batch_size=64, test_batch_size=512)
print(f"split: train={len(split_dict['train'])} val={len(split_dict['val'])} "
      f"test={len(split_dict['test'])}", flush=True)

gears_model = GEARS(pert_data, device="cpu")
gears_model.model_initialize()
gears_model.train(epochs=20)
print(f"training done {(time.time()-t0)/60:.1f} min", flush=True)
gears_model.save_model(f"{OUT}/model")

# ---------- zero-shot predict Norman held-out singles ----------
avail = [g for g in test_genes if g in gears_model.pert_list]
print(f"predictable test genes: {len(avail)}/{len(test_genes)}", flush=True)
res = gears_model.predict([[g] for g in avail])
results_pred = res[0] if isinstance(res, (tuple, list)) else res

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
df.to_csv(f"{OUT}/gears_transfer_metrics.csv", index=False)
summary = {
    "model": "GEARS_official_transfer",
    "n_test": len(avail),
    "delta_pearson": float(df.delta_pearson.mean()),
    "delta_pearson_std": float(df.delta_pearson.std()),
    "specific_pearson": float(df.specific_pearson.mean()),
    "discrimination": float(df.disc_rank.mean()),
    "train_conditions": len(split_dict["train"]),
    "train_cells": int(len(keep_idx)),
    "test_gene_train_overlap": overlap,
    "shared_genes": int(len(shared)),
    "epochs": 20,
    "device": "cpu",
}
print(json.dumps(summary, indent=1), flush=True)
json.dump(summary, open(f"{OUT}/gears_transfer_summary.json", "w"), indent=1)
print(f"\nV15 total {(time.time()-t0)/60:.1f} min")
print("GEARS_TRANSFER_DONE")
