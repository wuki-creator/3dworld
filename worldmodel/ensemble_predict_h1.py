# -*- coding: utf-8 -*-
"""Ensemble prediction: average raw deltas of N member ckpts, tight-decode, write h5ad.

Mirrors v14_predict.py's output format and decode recipe.
usage: python3 ensemble_predict_h1.py --ckpts a.pt b.pt ... --controls-dir D
        --genes gene_names.csv --perts pert_counts.csv [--val-signatures V]
        [--scale 1.0] [--amp 1.0] [--clip 3.0] [--cells-per-target 400]
        [--decode-style tight] [--jitter-shape 200] [--seed 101] --out OUT.h5ad
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

WORK = "/nfs_beijing_os/zizhuo_vcc/work"
if WORK not in sys.path:
    sys.path.insert(0, WORK)

import v14_predict as V

pred3 = V.pred3
calibrate = getattr(V, "calibrate", None)
if calibrate is None:
    def calibrate(row, ti, top_k, scale, self_scale=1.0, max_delta=5.0):
        return pred3.calibrate_effect(row, int(ti), top_k, scale, self_scale, max_delta)

p = argparse.ArgumentParser()
p.add_argument("--ckpts", nargs="+", required=True)
p.add_argument("--scale", type=float, default=None)
p.add_argument("--top-k", type=int, default=-1)
p.add_argument("--self-scale", type=float, default=1.0)
p.add_argument("--max-delta", type=float, default=5.0)
p.add_argument("--amp", type=float, default=1.0)
p.add_argument("--clip", type=float, default=3.0)
p.add_argument("--decode-style", choices=("scaffold", "tight"), default="tight")
p.add_argument("--jitter-shape", type=float, default=200.0)
p.add_argument("--cells-per-target", type=int, default=400)
p.add_argument("--seed", type=int, default=101)
p.add_argument("--controls-dir", required=True)
p.add_argument("--genes", required=True)
p.add_argument("--perts", required=True)
p.add_argument("--val-signatures", default="")
p.add_argument("--out", required=True)
p.add_argument("--device", default="cuda")
a = p.parse_args()

device = torch.device(a.device)

def load_genes(path):
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if rows and rows[0] and any(k in rows[0][0].lower() for k in ("gene", "target")):
        rows = rows[1:]
    return [r[0] for r in rows if r]

genes = load_genes(a.genes)
gene_index = {g: i for i, g in enumerate(genes)}
targets = np.array(load_genes(a.perts))
target_indices = np.array([gene_index[t] for t in targets], dtype=np.int64)

members = [V.V14Member(path, device) for path in a.ckpts]
print("members loaded: %d" % len(members), flush=True)

def ensemble_raw(control_mean, tidx):
    out = None
    for m in members:
        r = m.raw_delta(control_mean, tidx)
        out = r if out is None else out + r
    return out / len(members)

t0 = time.time()
scale = a.scale
if a.val_signatures:
    v = np.load(a.val_signatures, allow_pickle=False)
    vt = v["targets"].astype(str)
    keep = np.array([t in gene_index for t in vt])
    ti_ok = np.array([gene_index[t] for t in vt[keep]], dtype=np.int64)
    ctrl = v["controls"].astype(np.float32)[keep].mean(0)
    raw = ensemble_raw(ctrl, ti_ok)
    eff = v["effects"].astype(np.float32)[keep]
    cos = V.cosine_row(raw, eff).mean()
    print("ensemble raw val cosine=%.4f" % cos, flush=True)
    if scale is None:
        best_c, best_s = -1.0, 1.0
        for s in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            cal = np.stack([calibrate(row, ti, a.top_k, s, a.self_scale, a.max_delta)
                            for row, ti in zip(raw, ti_ok)])
            c = float(V.cosine_row(cal, eff).mean())
            print("  scale=%.2f -> %.4f" % (s, c), flush=True)
            if c > best_c:
                best_c, best_s = c, s
        scale = best_s
        print("chosen scale=%.2f (val cosine %.4f)" % (scale, best_c), flush=True)

if scale is None:
    scale = 1.0

matrices, obs_targets, obs_contexts = [], [], []
for context_index, context in enumerate(("A", "B", "C")):
    controls = ad.read_h5ad(Path(a.controls_dir) / ("context_%s.h5ad" % context))
    try:
        rows = pred3.choose_rows(controls, a.cells_per_target, a.seed + context_index)
        base = pred3.aligned_sparse(controls.X[rows], pred3.adata_genes(controls), genes)
    finally:
        del controls
    gene_mean = np.asarray(base.mean(0)).ravel().astype(np.float64)
    control_mean = pred3.log_cp10k_mean(base)
    raw = ensemble_raw(control_mean, target_indices)
    cal = np.stack([calibrate(row, ti, a.top_k, scale, a.self_scale, a.max_delta)
                    for row, ti in zip(raw, target_indices)])
    for position, target in enumerate(targets):
        rng = np.random.default_rng(a.seed + 10000 * context_index + position)
        cal_pos = cal[position] * a.amp
        if a.decode_style == "tight":
            decoded = V.decode_tight(gene_mean, cal_pos, a.cells_per_target, rng,
                                     jitter_shape=a.jitter_shape, clip=a.clip)
        else:
            decoded = pred3.bayesian_decode(base, cal_pos, gene_mean, rng, 2.0)
        matrices.append(decoded)
        obs_targets.extend([target] * a.cells_per_target)
        obs_contexts.extend([context] * a.cells_per_target)
        if (position + 1) % 50 == 0:
            print("context=%s %d/%d (%.0fs)" % (context, position + 1, len(targets), time.time() - t0), flush=True)
    print("context %s done (%.0fs)" % (context, time.time() - t0), flush=True)

matrix = sp.vstack(matrices, format="csr", dtype=np.uint32)
obs = pd.DataFrame({"target_gene": obs_targets, "context": obs_contexts},
                   index=["ens_%06d" % i for i in range(matrix.shape[0])])
prediction = ad.AnnData(X=matrix, obs=obs,
                        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")))
prediction.obs["target_gene"] = prediction.obs["target_gene"].astype("category")
prediction.obs["context"] = prediction.obs["context"].astype("category")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
prediction.write_h5ad(a.out)
diag = {"ckpts": a.ckpts, "scale": scale, "amp": a.amp, "clip": a.clip,
        "rows": int(matrix.shape[0])}
Path(str(a.out) + ".diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
print("wrote %s (%.0fs)" % (a.out, time.time() - t0), flush=True)
