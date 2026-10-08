# -*- coding: utf-8 -*-
"""Cross-member ensemble prediction for VCC 2026.

Members (LFC-signature level, official 18,533-gene panel):
  * MagWorld v13 checkpoints (H1-trained zero-shot CRISPRi world models)
  * MagWorld v4 checkpoints (H1-trained, published recipe family)
  * MagneticCrossCellV2 fine-tuned from a sci-Plex pretrain (drug slot bridged
    to target-gene embeddings)

Each member's raw delta is calibrated with its own recipe, weighted
(validation-optimised, non-negative, sum to 1), averaged, then decoded to raw
counts with the shared Gamma-Poisson decoder from predict_magworld_vcc2026_v3.
"""
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

MAGWORLD_SRC = "/home/zizhuo/maglab_deploy/src"
CROSS_SRC = "/home/zizhuo/cross_cell_vcc_v2/src"
for p in (MAGWORLD_SRC, CROSS_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

PRED3_PATH = "/home/zizhuo/maglab_deploy/src/predict_magworld_vcc2026_v3.py"
spec = importlib.util.spec_from_file_location("pred3", PRED3_PATH)
pred3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pred3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v13", nargs="*", default=[])
    p.add_argument("--v4", nargs="*", default=[])
    p.add_argument("--mccv2", nargs="*", default=[])
    p.add_argument("--v13-scale", type=float, default=0.5)
    p.add_argument("--v4-scale", type=float, default=2.0)
    p.add_argument("--mccv2-scale", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--self-scale", type=float, default=1.0)
    p.add_argument("--max-delta", type=float, default=5.0)
    p.add_argument("--prior-strength", type=float, default=2.0)
    p.add_argument("--cells-per-target", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
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


class MagWorldMember(object):
    """v13 or v4 checkpoint member."""

    def __init__(self, path, device, group):
        self.path = path
        self.group = group
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.genes = [str(g) for g in ck["genes"]]
        config = dict(ck["model_config"])
        errors = []
        model = None
        for module_name, class_name in (
            ("model_world_h1_v13", "WorldModelH1V13"),
            ("model_world_h1_v4", "WorldModelH1V4"),
            ("model_world_h1_v3", "WorldModelH1V3"),
        ):
            try:
                cls = getattr(importlib.import_module(module_name), class_name)
                candidate = cls(**config)
                candidate.load_state_dict(ck["model_state"])
                model = candidate
                self.kind = module_name
                break
            except Exception as exc:  # try the next architecture
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


class MCCV2Member(object):
    """Fine-tuned MagneticCrossCellV2 member (drug slot = target gene)."""

    def __init__(self, path, device):
        self.path = path
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ck["model_config"])
        self.genes = [str(g) for g in ck["genes"]]
        self.targets = [str(t) for t in ck["targets"]]
        self.gene_rows = np.asarray(ck["gene_rows"], dtype=np.int64)
        self.loops = int(ck.get("loops", 20))
        from magnetic_cross_cell_v2 import MagneticCrossCellV2
        self.model = MagneticCrossCellV2(**cfg).to(device).eval()
        self.model.load_state_dict(ck["model_state"])
        self.device = device
        emb = self.model.genes.weight.detach().cpu().numpy()
        self.emb_lookup = emb[self.gene_rows + 1]

    def raw_delta(self, control_mean, target_indices):
        control = torch.from_numpy(control_mean).to(self.device)
        drug_ids = []
        for t in target_indices:
            gene = self.genes[int(t)]
            if gene in self.targets:
                drug_ids.append(self.targets.index(gene) + 1)
            else:
                vec = self.model.genes.weight.detach().cpu().numpy()[int(t) + 1]
                sim = self.emb_lookup @ vec
                drug_ids.append(int(np.argmax(sim)) + 1)
        gene_ids = torch.from_numpy(target_indices.astype(np.int64) + 1).to(self.device)
        out = np.zeros((len(target_indices), len(control_mean)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(target_indices), 64):
                stop = min(start + 64, len(target_indices))
                x = control[None, :].expand(stop - start, -1)
                g = gene_ids[start:stop]
                dd = torch.tensor(drug_ids[start:stop], device=self.device)
                dose = torch.ones(stop - start, device=self.device)
                pred, _ = self.model.rollout(x, g, dd, dose, self.loops)
                out[start:stop] = pred.cpu().numpy().reshape(stop - start, out.shape[1])
        return out


def calibrate(delta, target_index, top_k, scale, self_scale, max_delta):
    return pred3.calibrate_effect(delta, int(target_index), top_k, scale, self_scale, max_delta)


def main():
    a = parse_args()
    t0 = time.time()
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    genes = pred3.read_genes(a.genes)
    gene_lookup = {g: i for i, g in enumerate(genes)}
    with open(a.perts, newline="", encoding="utf-8") as fh:
        targets = [row["target_gene"] for row in csv.DictReader(fh)]
    target_indices = np.asarray([gene_lookup[t] for t in targets], dtype=np.int64)

    members = []
    for path in a.v13:
        members.append((MagWorldMember(path, device, "v13"), a.v13_scale, a.top_k, "v13:" + Path(path).stem))
    for path in a.v4:
        members.append((MagWorldMember(path, device, "v4"), a.v4_scale, a.top_k, "v4:" + Path(path).stem))
    for path in a.mccv2:
        members.append((MCCV2Member(path, device), a.mccv2_scale, -1, "mccv2:" + Path(path).stem))
    if not members:
        raise SystemExit("no members provided")
    for m, _, _, name in members:
        if m.genes != genes:
            raise ValueError("member %s gene list differs from official panel" % name)
    print("members: %s" % ", ".join(n for _, _, _, n in members), flush=True)

    def member_deltas(control_mean, tidx):
        out = []
        for m, scale, top_k, name in members:
            raw = m.raw_delta(control_mean, tidx)
            cal = np.stack([calibrate(row, ti, top_k, scale, a.self_scale, a.max_delta)
                            for row, ti in zip(raw, tidx)])
            out.append(cal)
        return out

    # ---- validation-driven ensemble weights ----
    weights = np.full(len(members), 1.0 / len(members))
    weight_source = "equal"
    if a.val_signatures:
        v = np.load(a.val_signatures, allow_pickle=False)
        v_targets = v["targets"].astype(str)
        ok = np.array([t in gene_lookup for t in v_targets])
        v_tidx = np.asarray([gene_lookup[t] for t in v_targets[ok]], dtype=np.int64)
        v_eff = v["effects"].astype(np.float32)[ok]
        v_ctrl = v["controls"].astype(np.float32)[ok].mean(0)
        deltas = member_deltas(v_ctrl, v_tidx)
        cos_solo = [float(cosine_row(d, v_eff).mean()) for d in deltas]
        rngw = np.random.default_rng(7)
        stacked = np.stack(deltas)
        best_score = float(cosine_row(stacked.mean(0), v_eff).mean())
        best_w = weights.copy()
        for _ in range(4000):
            w = rngw.dirichlet(np.ones(len(members)))
            score = float(cosine_row(np.tensordot(w, stacked, axes=1), v_eff).mean())
            if score > best_score:
                best_score = score
                best_w = w
        weights = best_w
        weight_source = "validation_cosine=%.4f" % best_score
        print("solo val cosine: %s" % json.dumps(
            {n: round(s, 4) for s, (_, _, _, n) in zip(cos_solo, members)}), flush=True)
        print("ensemble weights %s (%s)" % (np.round(weights, 3).tolist(), weight_source), flush=True)

    # ---- per-context prediction and decode ----
    matrices = []
    obs_targets = []
    obs_contexts = []
    diagnostics = {}
    for context_index, context in enumerate(("A", "B", "C")):
        controls = ad.read_h5ad(Path(a.controls_dir) / ("context_%s.h5ad" % context))
        try:
            rows = pred3.choose_rows(controls, a.cells_per_target, a.seed + context_index)
            base = pred3.aligned_sparse(controls.X[rows], pred3.adata_genes(controls), genes)
        finally:
            del controls
        gene_mean = np.asarray(base.mean(0)).ravel().astype(np.float64)
        control_mean = pred3.log_cp10k_mean(base)
        deltas = member_deltas(control_mean, target_indices)
        ensemble = np.tensordot(weights, np.stack(deltas), axes=1).astype(np.float32)
        for position, target in enumerate(targets):
            rng = np.random.default_rng(a.seed + 10000 * context_index + position)
            decoded = pred3.bayesian_decode(base, ensemble[position], gene_mean, rng, a.prior_strength)
            matrices.append(decoded)
            obs_targets.extend([target] * a.cells_per_target)
            obs_contexts.extend([context] * a.cells_per_target)
            if (position + 1) % 50 == 0:
                print("context=%s %d/%d (%.0fs)" % (context, position + 1, len(targets), time.time() - t0), flush=True)
        diagnostics[context] = {"effect_l1_mean": float(np.mean(np.abs(ensemble)))}
        print("context %s done (%.0fs)" % (context, time.time() - t0), flush=True)

    matrix = sp.vstack(matrices, format="csr", dtype=np.uint32)
    obs = pd.DataFrame(
        {"target_gene": obs_targets, "context": obs_contexts},
        index=["ens_%06d" % i for i in range(matrix.shape[0])],
    )
    prediction = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes, name="gene_name")),
    )
    prediction.obs["target_gene"] = prediction.obs["target_gene"].astype("category")
    prediction.obs["context"] = prediction.obs["context"].astype("category")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(a.out)
    diag_path = str(a.out) + ".diagnostics.json"
    with open(diag_path, "w") as fh:
        json.dump({"members": [n for _, _, n in members],
                   "weights": weights.tolist(), "weight_source": weight_source,
                   "diagnostics": diagnostics, "seconds": time.time() - t0}, fh, indent=2)
    print("wrote %s (%.0fs)" % (a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
