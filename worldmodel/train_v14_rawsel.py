# -*- coding: utf-8 -*-
"""Train MagWorld v14 (physics-guided magnetic-flow) on H1 signatures.
Loss (P0-4): L1 + lambda_mmd * MMD + lambda_bce * BCE(sign)
             + lambda_kernel * ||kernel_alpha - physics_init||^2
Early stopping on the shared validation objective (keep-best)."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--signatures", required=True)
    p.add_argument("--official-perts", required=True)
    p.add_argument("--gene-embeddings", required=True)
    p.add_argument("--magworld-src", required=True)
    p.add_argument("--model-module", default="model_world_h1_v14")
    p.add_argument("--model-class", default="WorldModelH1V14")
    p.add_argument("--out", required=True)
    p.add_argument("--select-raw", action="store_true", default=False,
                   help="select checkpoint by raw validation cosine instead of swept objective")
    p.add_argument("--holdout-mode", choices=("official-overlap", "random", "none"),
                   default="official-overlap")
    p.add_argument("--random-holdout", type=int, default=50)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--lambda-l1", type=float, default=0.2)
    p.add_argument("--lambda-mmd", type=float, default=0.0)
    p.add_argument("--lambda-bce", type=float, default=0.15,
                   help="direction (sign-BCE) loss weight")
    p.add_argument("--lambda-kernel", type=float, default=0.02)
    p.add_argument("--lambda-shared-bias", type=float, default=0.05)
    p.add_argument("--bce-temperature", type=float, default=0.05)
    p.add_argument("--lambda-de", type=float, default=1.0)
    p.add_argument("--lambda-cosine", type=float, default=0.5)
    p.add_argument("--lambda-rank", type=float, default=0.1)
    p.add_argument("--lambda-pds", type=float, default=0.05)
    p.add_argument("--lambda-yield", type=float, default=0.15)
    p.add_argument("--rank-temperature", type=float, default=0.05)
    p.add_argument("--pds-tau", type=float, default=0.15)
    p.add_argument("--yield-threshold", type=float, default=0.05)
    p.add_argument("--yield-temperature", type=float, default=0.02)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--d-dir", type=int, default=32)
    p.add_argument("--graph-k", type=int, default=50)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--response-rank", type=int, default=64)
    p.add_argument("--max-delta", type=float, default=1.5)
    p.add_argument("--context-strength", type=float, default=0.25)
    p.add_argument("--langevin-init", type=float, default=4.0)
    p.add_argument("--kernel-powers", type=int, default=4)
    p.add_argument("--use-dipole", action="store_true", default=True)
    p.add_argument("--no-use-dipole", dest="use_dipole", action="store_false")
    p.add_argument("--use-langevin", action="store_true", default=True)
    p.add_argument("--no-use-langevin", dest="use_langevin", action="store_false")
    p.add_argument("--seed", type=int, default=421)
    p.add_argument("--device", default="cuda")
    p.add_argument("--init-from", default=None,
                   help="optional checkpoint to warm-start base-pathway weights")
    p.add_argument("--freeze-base", action="store_true",
                   help="freeze v4 base pathway; train only magnetic residual")
    return p.parse_args()


def gaussian_mmd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def sq_dist(x, y):
        d2 = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * (x @ y.T)
        return d2.clamp_min(0.0)  # fp32 cancellation guard

    d2 = torch.cat([sq_dist(a, a), sq_dist(b, b)], 0)
    med = d2.flatten().median().clamp_min(1e-8)
    sigma2 = med / 2.0
    kaa = torch.exp(-sq_dist(a, a) / (2 * sigma2))
    kbb = torch.exp(-sq_dist(b, b) / (2 * sigma2))
    kab = torch.exp(-sq_dist(a, b) / (2 * sigma2))
    m = a.shape[0]
    n = b.shape[0]
    if m < 2 or n < 2:
        return a.new_zeros(())
    return ((kaa.sum() - kaa.diag().sum()) / (m * (m - 1))
            + (kbb.sum() - kbb.diag().sum()) / (n * (n - 1))
            - 2.0 * kab.mean())


def sign_bce(prediction, truth, mask, temperature):
    logits = prediction / temperature
    targets = (torch.sign(truth) > 0).float()
    weights = (truth.abs() * mask)
    weights = weights / weights.sum().clamp_min(1e-6)
    per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (per * weights).sum()


def cosine_loss(pred, truth, weights):
    num = (pred * truth).sum(1)
    den = pred.norm(dim=1) * truth.norm(dim=1) + 1e-8
    return ((1.0 - num / den) * weights).mean()


def de_loss(pred, truth, mask, weights):
    per = ((pred - truth).abs() * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return (per * weights).mean()


def rank_loss(pred, truth, top_k, temperature):
    idx = truth.abs().topk(min(top_k, truth.shape[1]), dim=1).indices
    p = pred.gather(1, idx).abs()
    t = truth.gather(1, idx).abs()
    pdiff = p[:, :, None] - p[:, None, :]
    tsign = (t[:, :, None] - t[:, None, :]).sign()
    pair = F.softplus(-pdiff * tsign / temperature)
    return pair.mean()


def pds_loss(pred, tau):
    z = F.normalize(pred, dim=1)
    sim = (z @ z.T) / tau
    target = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(sim, target)


def yield_loss(pred, truth, threshold, temperature):
    fp = torch.sigmoid((pred.abs() - threshold) / temperature).mean(1)
    ft = (truth.abs() > threshold).float().mean(1)
    return (fp - ft).pow(2).mean()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    src = str(Path(args.magworld_src).resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    t4 = importlib.import_module("train_magworld_h1_v4")

    dataset = np.load(args.signatures, allow_pickle=False)
    genes = dataset["genes"].astype(str).tolist()
    targets = dataset["targets"].astype(str)
    controls = dataset["controls"].astype(np.float32)
    effects = dataset["effects"].astype(np.float32)
    replicate_controls = dataset["replicate_controls"].astype(np.float32)
    replicate_effects = dataset["replicate_effects"].astype(np.float32)
    replicate_targets = dataset["replicate_targets"].astype(np.int64)
    replicate_cells = dataset["replicate_cells"].astype(np.float32)
    features = np.load(args.gene_embeddings).astype(np.float32)
    if features.shape[0] != len(genes):
        raise ValueError("gene embeddings and signature genes differ")
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    gene_lookup = {g: i for i, g in enumerate(genes)}
    target_gene_indices = np.asarray([gene_lookup[t] for t in targets])
    official = set(pd.read_csv(args.official_perts)["target_gene"].astype(str))

    if args.holdout_mode == "official-overlap":
        validation_indices = np.asarray(
            [i for i, t in enumerate(targets) if t in official], dtype=np.int64)
    elif args.holdout_mode == "random":
        validation_indices = np.sort(
            rng.choice(len(targets), min(args.random_holdout, len(targets) // 3), replace=False))
    else:
        validation_indices = np.asarray([], dtype=np.int64)
    validation_set = set(validation_indices.tolist())
    train_indices = np.asarray(
        [i for i in range(len(targets)) if i not in validation_set], dtype=np.int64)
    if len(validation_indices) == 0:
        validation_indices = train_indices
    train_set = set(train_indices.tolist())
    replicate_train = np.asarray(
        [i for i, t in enumerate(replicate_targets) if int(t) in train_set], dtype=np.int64)

    validation_target_genes = target_gene_indices[validation_indices]
    baseline = {
        "zero": {"top_k": -1, "scale": 0.0,
                 "metrics": t4.signature_metrics(
                     np.zeros_like(effects[validation_indices]),
                     effects[validation_indices], validation_target_genes)},
        "mean": t4.calibrate(
            np.repeat(effects[train_indices].mean(0, keepdims=True), len(validation_indices), axis=0),
            effects[validation_indices], validation_target_genes),
        "ridge": t4.ridge_baseline(
            features[target_gene_indices], effects, train_indices,
            validation_indices, validation_target_genes),
        "nearest_neighbor": t4.nearest_neighbor_baseline(
            features[target_gene_indices], effects, train_indices,
            validation_indices, validation_target_genes),
    }
    baseline["zero"]["objective"] = t4.metric_objective(baseline["zero"]["metrics"])
    print(json.dumps({"baselines": baseline}), flush=True)

    model_module = importlib.import_module(args.model_module)
    model_class = getattr(model_module, args.model_class)
    model_config = {
        "n_genes": len(genes),
        "d_model": features.shape[1],
        "d_z": args.latent_dim,
        "d_hidden": args.hidden_dim,
        "d_dir": args.d_dir,
        "max_delta": args.max_delta,
        "context_strength": args.context_strength,
        "shared_bias_mode": "gated",
        "shared_bias_initial_scale": 0.1,
        "normalize_context": True,
        "graph_k": args.graph_k,
        "n_layers": args.n_layers,
        "response_rank": args.response_rank,
        "use_dipole": args.use_dipole,
        "use_langevin": args.use_langevin,
        "langevin_init": args.langevin_init,
        "kernel_powers": args.kernel_powers,
    }
    model = model_class(**model_config).to(device)
    model.initialize_gene_embeddings(torch.from_numpy(features).to(device))
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
        print(json.dumps({
            "init_from": args.init_from,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }), flush=True)
        with torch.no_grad():
            model.amp_bias.fill_(-4.0)  # start magnetic pathway nearly closed
        if args.freeze_base:
            frozen = 0
            for name, param in model.named_parameters():
                if name.startswith(("context_encoder", "context_receiver",
                                    "gene_bias", "raw_shared_bias_scale")):
                    param.requires_grad_(False)
                    frozen += 1
            print(json.dumps({"frozen_base_params": frozen}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    canonical_abs = np.abs(effects)
    canonical_top = np.argpartition(canonical_abs, -args.top_k, axis=1)[:, -args.top_k:]
    de_masks = np.zeros_like(effects, dtype=bool)
    np.put_along_axis(de_masks, canonical_top, True, axis=1)
    sample_weight = np.sqrt(replicate_cells[replicate_train].clip(min=1.0))
    sample_weight /= sample_weight.mean()
    alpha_init = model.kernel_alpha_init.to(device)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_objective = -math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(replicate_train))
        totals = {"loss": 0.0, "l1": 0.0, "mmd": 0.0, "bce": 0.0, "de": 0.0,
                  "cos": 0.0, "rank": 0.0, "pds": 0.0, "yield": 0.0,
                  "kernel": 0.0, "shared_bias": 0.0}
        batches = 0
        for start in range(0, len(order), args.batch_size):
            positions = order[start:start + args.batch_size]
            sample_indices = replicate_train[positions]
            target_indices = replicate_targets[sample_indices]
            xb = torch.from_numpy(replicate_controls[sample_indices]).to(device)
            yb = torch.from_numpy(replicate_effects[sample_indices]).to(device)
            pert = torch.from_numpy(target_gene_indices[target_indices]).to(device)
            mask = torch.from_numpy(de_masks[target_indices]).to(device)
            weights = torch.from_numpy(sample_weight[positions]).to(device)

            prediction = model.predict_delta(xb, pert)
            l1_per = (prediction - yb).abs().mean(1)
            l1 = (l1_per * weights).mean()
            mmd = gaussian_mmd(prediction, yb) if args.lambda_mmd > 0 else l1.new_zeros(())
            bce = (sign_bce(prediction, yb, mask, args.bce_temperature)
                   if args.lambda_bce > 0 else l1.new_zeros(()))
            de = (de_loss(prediction, yb, mask, weights)
                  if args.lambda_de > 0 else l1.new_zeros(()))
            cos = (cosine_loss(prediction, yb, weights)
                   if args.lambda_cosine > 0 else l1.new_zeros(()))
            rnk = (rank_loss(prediction, yb, args.top_k, args.rank_temperature)
                   if args.lambda_rank > 0 else l1.new_zeros(()))
            pds = (pds_loss(prediction, args.pds_tau)
                   if args.lambda_pds > 0 and prediction.shape[0] > 1 else l1.new_zeros(()))
            yld = (yield_loss(prediction, yb, args.yield_threshold, args.yield_temperature)
                   if args.lambda_yield > 0 else l1.new_zeros(()))
            kernel_prior = ((model.kernel_alpha().to(device) - alpha_init) ** 2).sum()
            shared_bias = model.effective_shared_bias_scale() * model.gene_bias.abs().mean()
            loss = (args.lambda_l1 * l1 + args.lambda_mmd * mmd + args.lambda_bce * bce
                    + args.lambda_de * de + args.lambda_cosine * cos
                    + args.lambda_rank * rnk + args.lambda_pds * pds
                    + args.lambda_yield * yld
                    + args.lambda_kernel * kernel_prior + args.lambda_shared_bias * shared_bias)
            if not torch.isfinite(loss):
                import numpy as _np
                _np.savez("/nfs_beijing_os/zizhuo_vcc/work/nan_dump.npz",
                          xb=xb.detach().cpu().numpy(), yb=yb.cpu().numpy(),
                          pert=pert.cpu().numpy())
                raise RuntimeError(
                    "nonfinite loss at epoch %d: l1=%r mmd=%r bce=%r kernel=%r shared=%r "
                    "pred_finite=%r y_finite=%r x_finite=%r"
                    % (epoch, float(l1), float(mmd), float(bce), float(kernel_prior),
                       float(shared_bias), bool(torch.isfinite(prediction).all()),
                       bool(torch.isfinite(yb).all()), bool(torch.isfinite(xb).all())))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in {"loss": loss, "l1": l1, "mmd": mmd, "bce": bce,
                               "de": de, "cos": cos, "rank": rnk, "pds": pds,
                               "yield": yld,
                               "kernel": kernel_prior, "shared_bias": shared_bias}.items():
                totals[key] += float(value.detach())
            batches += 1

        model.eval()
        with torch.no_grad():
            val_chunks = []
            for s in range(0, len(validation_indices), 8):
                e = min(s + 8, len(validation_indices))
                vp = model.predict_delta(
                    torch.from_numpy(controls[validation_indices[s:e]]).to(device),
                    torch.from_numpy(validation_target_genes[s:e]).to(device)).cpu().numpy()
                val_chunks.append(vp)
            val_pred = np.concatenate(val_chunks, axis=0)
        calibration = t4.calibrate(val_pred, effects[validation_indices], validation_target_genes)
        val_truth = effects[validation_indices]
        raw_cosine = float(np.mean(
            np.sum(val_pred * val_truth, axis=1)
            / np.maximum(
                np.linalg.norm(val_pred, axis=1) * np.linalg.norm(val_truth, axis=1),
                1e-8,
            )
        ))
        calibration["raw_cosine"] = raw_cosine
        summary = {
            "epoch": epoch,
            "training": {k: v / max(1, batches) for k, v in totals.items()},
            "validation": calibration,
            "magnetic": model.magnetic_diagnostics(
                torch.from_numpy(validation_target_genes[:min(16, len(validation_target_genes))]).to(device)),
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)
        objective = float(calibration["objective"])
        if args.select_raw:
            objective = raw_cosine
            improved = objective > best_objective
        else:
            improved = objective > best_objective or args.holdout_mode == "none"
        if improved:
            best_objective = objective
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model_config,
                "genes": genes,
                "features_path": args.gene_embeddings,
                "signature_targets": targets.tolist(),
                "signature_effects": effects,
                "signature_controls": controls,
                "train_targets": targets[train_indices].tolist(),
                "validation_targets": targets[validation_indices].tolist(),
                "baseline": baseline,
                "best_epoch": best_epoch,
                "best_validation": calibration,
                "checkpoint_selection": ("raw_cosine" if args.select_raw else
                                          ("last_epoch" if args.holdout_mode == "none"
                                          else "validation_objective")),
                "initialization": None,
                "history": history,
                "training_args": vars(args),
            }
            temporary = output.with_suffix(output.suffix + ".tmp")
            torch.save(checkpoint, temporary, pickle_protocol=4)
            os.replace(temporary, output)
            output.with_suffix(".json").write_text(
                json.dumps({k: v for k, v in checkpoint.items()
                            if k not in {"model_state", "signature_effects", "signature_controls"}},
                           indent=2) + "\n", encoding="utf-8")
        else:
            stale_epochs += 1
        if args.holdout_mode != "none" and stale_epochs >= args.patience:
            print(json.dumps({"early_stop": epoch, "best_epoch": best_epoch,
                              "best_objective": best_objective}), flush=True)
            break
    print(json.dumps({"checkpoint": str(output), "best_epoch": best_epoch,
                      "best_objective": best_objective}), flush=True)


if __name__ == "__main__":
    main()
