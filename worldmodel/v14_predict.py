# -*- coding: utf-8 -*-
"""Single-member VCC submission prediction with the v14 magnetic-flow model."""
import argparse
import csv
import importlib
import importlib.util
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
MAGWORLD_SRC = "/home/zizhuo/maglab_deploy/src"
for p in (WORK, MAGWORLD_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

PRED3_PATH = MAGWORLD_SRC + "/predict_magworld_vcc2026_v3.py"
spec = importlib.util.spec_from_file_location("pred3", PRED3_PATH)
pred3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pred3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scale", type=float, default=None,
                   help="fixed calibration scale (default: sweep on validation)")
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--self-scale", type=float, default=1.0)
    p.add_argument("--max-delta", type=float, default=5.0)
    p.add_argument("--prior-strength", type=float, default=2.0)
    p.add_argument("--decode-style", choices=("scaffold", "tight"), default="scaffold",
                   help="scaffold: modify real control cells; tight: low-dispersion "
                        "Poisson cells around the predicted mean")
    p.add_argument("--jitter-shape", type=float, default=200.0,
                   help="Gamma shape for per-cell count jitter in tight mode "
                        "(larger = tighter; 1e6 ~ pure Poisson)")
    p.add_argument("--cells-per-target", type=int, default=400)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--controls-dir", required=True)
    p.add_argument("--genes", required=True)
    p.add_argument("--perts", required=True)
    p.add_argument("--val-signatures", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def cosine_row(pred, truth):
    num = (pred * truth).sum(1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    return num / np.maximum(den, 1e-8)


class V14Member(object):
    def __init__(self, path, device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.genes = [str(g) for g in ck["genes"]]
        config = dict(ck["model_config"])
        model = None
        errors = []
        for module_name, class_name in (
            ("model_world_h1_v14", "WorldModelH1V14"),
            ("model_world_h1_v13", "WorldModelH1V13"),
            ("model_world_h1_v4", "WorldModelH1V4"),
        ):
            try:
                cls = getattr(importlib.import_module(module_name), class_name)
                candidate = cls(**config)
                candidate.load_state_dict(ck["model_state"])
                model = candidate
                self.kind = module_name
                break
            except Exception as exc:
                errors.append("%s: %s" % (module_name, exc))
        if model is None:
            raise RuntimeError("no architecture fit %s: %s" % (path, errors))
        self.model = model.to(device).eval()
        self.device = device

    def raw_delta(self, control_mean, target_indices):
        control = torch.from_numpy(control_mean).to(self.device)
        out = np.zeros((len(target_indices), len(control_mean)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(target_indices), 64):
                stop = min(start + 64, len(target_indices))
                idx = torch.from_numpy(target_indices[start:stop]).to(self.device)
                controls = control[None, :].expand(stop - start, -1)
                out[start:stop] = self.model.predict_delta(controls, idx).cpu().numpy()
        return out


def calibrate(delta, target_index, top_k, scale, self_scale, max_delta):
    return pred3.calibrate_effect(delta, int(target_index), top_k, scale, self_scale, max_delta)


def decode_tight(gene_mean, delta, n_cells, rng, jitter_shape=200.0):
    """Low-dispersion decode: Poisson cells around the predicted perturbed mean.

    gene_mean: per-gene expected counts from the context controls.
    delta: predicted LFC vector.
    jitter_shape: Gamma shape for per-cell total-count jitter (200 => ~7% CV).
    """
    mu = np.maximum(gene_mean, 1e-4) * np.exp(np.clip(delta, -8.0, 8.0))
    scale = rng.gamma(shape=jitter_shape, scale=1.0 / jitter_shape, size=(n_cells, 1))
    counts = rng.poisson(mu[None, :] * scale)
    return sp.csr_matrix(counts.astype(np.uint32))


def main():
    a = parse_args()
    t0 = time.time()
    device = torch.device(a.device)
    genes = pred3.read_genes(a.genes)
    gene_lookup = {g: i for i, g in enumerate(genes)}
    with open(a.perts, newline="", encoding="utf-8") as fh:
        targets = [row["target_gene"] for row in csv.DictReader(fh)]
    target_indices = np.asarray([gene_lookup[t] for t in targets], dtype=np.int64)

    member = V14Member(a.ckpt, device)
    if member.genes != genes:
        raise ValueError("checkpoint gene list differs from official panel")
    print("loaded %s (%.0fs)" % (a.ckpt, time.time() - t0), flush=True)

    scale = a.scale
    if scale is None:
        if not a.val_signatures:
            raise SystemExit("need --val-signatures for scale sweep or pass --scale")
        v = np.load(a.val_signatures, allow_pickle=False)
        v_targets = v["targets"].astype(str)
        ok = np.array([t in gene_lookup for t in v_targets])
        v_tidx = np.asarray([gene_lookup[t] for t in v_targets[ok]], dtype=np.int64)
        v_eff = v["effects"].astype(np.float32)[ok]
        v_ctrl = v["controls"].astype(np.float32)[ok].mean(0)
        raw = member.raw_delta(v_ctrl, v_tidx)
        print("raw val cosine=%.4f" % cosine_row(raw, v_eff).mean(), flush=True)
        best_s, best_c = None, -1.0
        for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            cal = np.stack([calibrate(row, ti, a.top_k, s, a.self_scale, a.max_delta)
                            for row, ti in zip(raw, v_tidx)])
            c = float(cosine_row(cal, v_eff).mean())
            print("  scale=%.2f -> %.4f" % (s, c), flush=True)
            if c > best_c:
                best_c, best_s = c, s
        scale = best_s
        print("chosen scale=%.2f (val cosine %.4f)" % (scale, best_c), flush=True)

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
        raw = member.raw_delta(control_mean, target_indices)
        cal = np.stack([calibrate(row, ti, a.top_k, scale, a.self_scale, a.max_delta)
                        for row, ti in zip(raw, target_indices)])
        for position, target in enumerate(targets):
            rng = np.random.default_rng(a.seed + 10000 * context_index + position)
            if a.decode_style == "tight":
                decoded = decode_tight(gene_mean, cal[position], a.cells_per_target, rng,
                                       jitter_shape=a.jitter_shape)
            else:
                decoded = pred3.bayesian_decode(base, cal[position], gene_mean, rng, a.prior_strength)
            matrices.append(decoded)
            obs_targets.extend([target] * a.cells_per_target)
            obs_contexts.extend([context] * a.cells_per_target)
            if (position + 1) % 50 == 0:
                print("context=%s %d/%d (%.0fs)" % (context, position + 1, len(targets), time.time() - t0), flush=True)
        print("context %s done (%.0fs)" % (context, time.time() - t0), flush=True)

    matrix = sp.vstack(matrices, format="csr", dtype=np.uint32)
    obs = pd.DataFrame({"target_gene": obs_targets, "context": obs_contexts},
                       index=["v14_%06d" % i for i in range(matrix.shape[0])])
    prediction = ad.AnnData(X=matrix, obs=obs,
                            var=pd.DataFrame(index=pd.Index(genes, name="gene_name")))
    prediction.obs["target_gene"] = prediction.obs["target_gene"].astype("category")
    prediction.obs["context"] = prediction.obs["context"].astype("category")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(a.out)
    diag = {"ckpt": a.ckpt, "scale": scale, "rows": int(matrix.shape[0])}
    Path(str(a.out) + ".diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    print("wrote %s (%.0fs)" % (a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
