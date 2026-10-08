# -*- coding: utf-8 -*-
"""Cell-level DE fid proxy: simulate the platform's DE test on decoded cells.

For each held-out val perturbation:
  - decode 400 perturbed cells (tight Poisson decode, amp x jitter grid)
  - compare against 400 real control cells with a per-gene Welch t-test
  - BH-correct, take significant genes, compare sign agreement with the true
    signature effect over the true top-100 DE genes

This mimics what the platform plausibly does to compute fid/jac/reach, so
decode-style / amplification / tightness can be optimized locally.

Usage:
  python3 cell_fid_proxy.py --ckpt <best.pt> [--amps 1,2,4] [--jitters 50,200,1000]
"""
import argparse
import importlib
import sys

import numpy as np
import torch

WORK = "/nfs_beijing_os/zizhuo_vcc/work"
MAGWORLD_SRC = "/home/zizhuo/maglab_deploy/src"
for p in (WORK, MAGWORLD_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

VAL = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz"
CONTROLS = "/home/zizhuo/vcc_data/extracted/context_A.h5ad"


def load_member(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    genes = [str(g) for g in ck["genes"]]
    config = dict(ck["model_config"])
    model = None
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
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError("no architecture fit %s" % path)
    return model.to(device).eval(), genes


def raw_delta(model, ctrl, tidx, device, batch=64):
    control = torch.from_numpy(ctrl).to(device)
    out = np.zeros((len(tidx), len(ctrl)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(tidx), batch):
            e = min(s + batch, len(tidx))
            idx = torch.from_numpy(tidx[s:e]).to(device)
            x = control[None, :].expand(e - s, -1)
            out[s:e] = model.predict_delta(x, idx).cpu().numpy()
    return out


def calibrate_like(row, ti, top_k, scale, self_scale=1.0, max_delta=5.0):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pred3", MAGWORLD_SRC + "/predict_magworld_vcc2026_v3.py")
    pred3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pred3)
    return pred3.calibrate_effect(row, int(ti), top_k, scale, self_scale, max_delta)


def decode_tight(gene_mean, delta, n_cells, rng, jitter_shape, clip=8.0):
    mu = np.maximum(gene_mean, 1e-4) * np.exp(np.clip(delta, -clip, clip))
    scale = rng.gamma(shape=jitter_shape, scale=1.0 / jitter_shape,
                      size=(n_cells, 1))
    return rng.poisson(mu[None, :] * scale).astype(np.float32)


def decode_scaffold(base_counts, delta, rng, prior_strength=2.0):
    """Modify real control cells (v13solo style)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pred3", MAGWORLD_SRC + "/predict_magworld_vcc2026_v3.py")
    pred3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pred3)
    gene_mean = np.maximum(np.asarray(base_counts.mean(0)).ravel(), 1e-8)
    dec = pred3.bayesian_decode(base_counts, delta, gene_mean, rng,
                                prior_strength)
    return (dec.toarray() if hasattr(dec, "toarray") else np.asarray(dec)).astype(np.float32)


def decode_hybrid(base_counts, gene_mean, delta, n_top, rng,
                  jitter_shape=200.0):
    """Keep real control cells but re-sample the top-|delta| genes from the
    predicted perturbed mean -> real dispersion + clean, powerful DE signal."""
    dec = base_counts.copy().astype(np.float32)
    n_cells = dec.shape[0]
    n_top = int(min(n_top, len(delta)))
    top = np.argpartition(np.abs(delta), -n_top)[-n_top:]
    mu = np.maximum(gene_mean[top], 1e-4) * np.exp(
        np.clip(delta[top], -8.0, 8.0))
    scale = rng.gamma(shape=jitter_shape, scale=1.0 / jitter_shape,
                      size=(n_cells, 1))
    dec[:, top] = rng.poisson(mu[None, :] * scale).astype(np.float32)
    return dec


def welch_ttest(a, b):
    """Vectorized two-sample Welch t-test. a,b: (n_cells, n_genes)."""
    from scipy import stats
    n1, n2 = a.shape[0], b.shape[0]
    m1, m2 = a.mean(0), b.mean(0)
    v1, v2 = a.var(0, ddof=1), b.var(0, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / np.maximum(se, 1e-12)
    # Welch-Satterthwaite df
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / max(n1 - 1, 1) + (v2 / n2) ** 2 / max(n2 - 1, 1)
    df = np.maximum(num / np.maximum(den, 1e-12), 1.0)
    p = 2 * stats.t.sf(np.abs(t), df)
    return t, p


def bh_fdr(p):
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out = np.empty(n)
    q_out[order] = np.minimum(q_sorted, 1.0)
    return q_out


def cell_fid_for_config(delta_cal, eff, ctrl_cells, gene_mean, mode, amp,
                        jitter, n_cells=400, top_k=100, seed=7,
                        ctrl_sparse=None, clip=8.0):
    """Return per-pert dict of DE-test agreement statistics."""
    rng = np.random.default_rng(seed)
    n_perts = len(delta_cal)
    base = ctrl_cells[:n_cells]
    stats_rows = []
    for i in range(n_perts):
        d = delta_cal[i] * amp
        if mode == "tight":
            dec = decode_tight(gene_mean, d, n_cells, rng, jitter, clip=clip)
        elif mode == "hybrid":
            dec = decode_hybrid(base, gene_mean, d, jitter, rng,
                                jitter_shape=200.0)
        else:
            dec = decode_scaffold(
                ctrl_sparse[:n_cells] if ctrl_sparse is not None else base,
                d, rng, prior_strength=jitter)
        # log1p like standard scRNA practice
        a = np.log1p(dec)
        bmat = np.log1p(base)
        t, p = welch_ttest(a, bmat)
        q = bh_fdr(np.nan_to_num(p, nan=1.0))
        detected = q < 0.05
        detected_sign = np.sign(t)
        # truth: top-k genes by |effect|
        truth_top = np.argpartition(np.abs(eff[i]), -top_k)[-top_k:]
        w = np.abs(eff[i][truth_top])
        truth_sign = np.sign(eff[i][truth_top])
        hit = detected[truth_top]
        agree = (detected_sign[truth_top] == truth_sign) & hit & (w > 0)
        power = hit.mean()
        fid_weighted = (w * agree).sum() / max(w.sum(), 1e-8)
        sign_acc = agree.sum() / max(hit.sum(), 1)
        # platform-guess: fidelity minus missed penalty (missing counts as -1)
        fid_pen = ((w * np.where(hit & (detected_sign[truth_top] == truth_sign), 1.0,
                                 -1.0)).sum() / max(w.sum(), 1e-8))
        # DE logFC magnitude accuracy on truth top genes (t-stat ~ logFC proxy)
        lfc_err = np.abs(t[truth_top] - eff[i][truth_top]).mean() / max(
            np.abs(eff[i][truth_top]).mean(), 1e-8)
        stats_rows.append(dict(power=power, fid_w=fid_weighted,
                               sign_acc=sign_acc, fid_pen=fid_pen,
                               lfc_nmae=float(lfc_err),
                               n_detected=float(detected.sum())))
    agg = {k: float(np.mean([r[k] for r in stats_rows]))
           for k in stats_rows[0]}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--amps", default="1.0")
    ap.add_argument("--jitters", default="200")
    ap.add_argument("--modes", default="tight,scaffold")
    ap.add_argument("--cells", type=int, default=400)
    ap.add_argument("--raw", action="store_true",
                    help="skip calibrate_effect; use raw model delta (clipped)")
    ap.add_argument("--clip", type=float, default=8.0,
                    help="fold-change clip (log) for tight decode")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = np.load(VAL, allow_pickle=False)
    model, genes = load_member(args.ckpt, device)
    lookup = {g: i for i, g in enumerate(genes)}
    targets = v["targets"].astype(str)
    ok = np.array([t in lookup for t in targets])
    tidx = np.asarray([lookup[t] for t in targets[ok]], dtype=np.int64)
    eff = v["effects"].astype(np.float32)[ok]
    ctrl = v["controls"].astype(np.float32)[ok].mean(0)

    raw = raw_delta(model, ctrl, tidx, device)
    if args.raw:
        cal = np.clip(raw, -5.0, 5.0)
        print("raw delta mode (no calibrate_effect); mean|delta|=%.5f max=%.3f"
              % (np.abs(raw).mean(), np.abs(raw).max()), flush=True)
    else:
        cal = np.stack([calibrate_like(r, ti, 500, 1.0) for r, ti in zip(raw, tidx)])

    import anndata as ad
    controls = ad.read_h5ad(CONTROLS)
    ctrl_sparse = controls.X[:args.cells]
    ctrl_cells = (ctrl_sparse.toarray() if hasattr(ctrl_sparse, "toarray")
                  else np.asarray(ctrl_sparse)).astype(np.float32)
    gene_mean = np.maximum(ctrl_cells.mean(0).astype(np.float64), 1e-8)

    print("n_val_perts=%d cells=%d" % (len(tidx), args.cells), flush=True)
    for mode in args.modes.split(","):
        for amp in [float(s) for s in args.amps.split(",")]:
            for jt in [float(s) for s in args.jitters.split(",")]:
                r = cell_fid_for_config(cal, eff, ctrl_cells, gene_mean,
                                        mode, amp, jt,
                                        n_cells=args.cells,
                                        ctrl_sparse=ctrl_sparse,
                                        clip=args.clip)
                print("mode=%-8s amp=%.2f jitter=%8.1f  power=%.4f fid_w=%.4f "
                      "sign_acc=%.4f fid_pen=%.4f lfc_nmae=%.3f n_det=%.0f"
                      % (mode, amp, jt, r["power"], r["fid_w"], r["sign_acc"],
                         r["fid_pen"], r["lfc_nmae"], r["n_detected"]), flush=True)


if __name__ == "__main__":
    main()
