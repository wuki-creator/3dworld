
# -*- coding: utf-8 -*-
"""Local proxy scorer for the six VCC platform metrics.

Implements delta-space proxies for fid / reach / jac / nmae plus a decode-based
perturbation-discrimination (pds) proxy, so calibration / decode parameters can
be swept offline without burning daily submission slots.

Usage:
  python3 proxy_score.py --ckpt <best.pt> [--ckpt2 ...] --scales 0.5,1,1.5,2,3
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


def delta_metrics(pred, truth, tidx, top_k=100):
    """fid / reach / jac / nmae at delta level."""
    n = pred.shape[0]
    fid_num = np.zeros(n)
    fid_den = np.zeros(n)
    reach = np.zeros(n)
    jac = np.zeros(n)
    nmae = np.zeros(n)
    arange = np.arange(pred.shape[1])
    for i in range(n):
        t = truth[i]
        p = pred[i]
        de = np.argpartition(np.abs(t), -top_k)[-top_k:]
        w = np.abs(t[de])
        agree = (np.sign(p[de]) == np.sign(t[de])) & (w > 0)
        fid_num[i] = (w * agree).sum()
        fid_den[i] = w.sum()
        pred_top = np.argpartition(np.abs(p), -top_k)[-top_k:]
        inter = len(np.intersect1d(de, pred_top))
        reach[i] = inter / top_k
        jac[i] = inter / (2 * top_k - inter)
        denom = np.abs(t).sum()
        nmae[i] = np.abs(p - t).sum() / max(denom, 1e-8)
    return dict(fid=float((fid_num / np.maximum(fid_den, 1e-8)).mean()),
                reach=float(reach.mean()), jac=float(jac.mean()),
                nmae=float(nmae.mean()))


def _decode_tight(gene_mean, delta, n_cells, rng, jitter_shape=200.0):
    """Poisson cells around predicted perturbed mean (copied from v14_predict)."""
    import scipy.sparse as sp
    mu = np.maximum(gene_mean, 1e-4) * np.exp(np.clip(delta, -8.0, 8.0))
    scale = rng.gamma(shape=jitter_shape, scale=1.0 / jitter_shape, size=(n_cells, 1))
    counts = rng.poisson(mu[None, :] * scale)
    return sp.csr_matrix(counts.astype(np.uint32))


def pds_proxy(delta, ctrl_counts, genes_mean, n_targets=30, cells=60, seed=3,
              jitter=200.0, mode="tight"):
    """Decode-based perturbation discrimination proxy (PCA + Mann-Whitney AUC).

    mode "tight": fresh Poisson decode per target with gamma jitter_shape=jitter.
    mode "bayesian": bayesian_decode with prior_strength=jitter.
    """
    import importlib.util
    rng = np.random.default_rng(seed)
    n_ctrl = int(ctrl_counts.shape[0])
    gene_mean_all = np.asarray(ctrl_counts.mean(0)).ravel().astype(np.float64)
    gene_mean_all = np.maximum(gene_mean_all, 1e-8)
    spec = None
    if mode == "bayesian":
        spec = importlib.util.spec_from_file_location(
            "pred3", MAGWORLD_SRC + "/predict_magworld_vcc2026_v3.py")
        pred3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pred3)
    pick = rng.choice(len(delta), size=min(n_targets, len(delta)), replace=False)
    mats = []
    labels = []
    for j, ti in enumerate(pick):
        if mode == "tight":
            dec = _decode_tight(gene_mean_all, delta[ti], cells, rng, jitter)
        else:
            if n_ctrl >= cells:
                start = int((j * cells) % (n_ctrl - cells + 1))
                base = ctrl_counts[start:start + cells]
            else:
                base = ctrl_counts
            gm = np.maximum(np.asarray(base.mean(0)).ravel().astype(np.float64), 1e-8)
            dec = pred3.bayesian_decode(base, delta[ti], gm, rng, jitter)
        arr = dec.toarray() if hasattr(dec, "toarray") else np.asarray(dec)
        arr = np.nan_to_num(np.log1p(arr.astype(np.float64)),
                            nan=0.0, posinf=0.0, neginf=0.0)
        mats.append(arr)
        labels.extend([j] * arr.shape[0])
    X = np.concatenate(mats, 0)
    labels = np.asarray(labels)
    X = X - X.mean(0, keepdims=True)
    X = np.nan_to_num(X, nan=0.0)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    Z = u[:, :20] * s[:20]
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    n = len(Z)
    sim = Z @ Z.T
    same_mask = labels[:, None] == labels[None, :]
    iu = np.triu_indices(n, 1)
    s_vals = sim[iu]
    s_same = s_vals[same_mask[iu]]
    s_diff = s_vals[~same_mask[iu]]
    if len(s_same) < 10 or len(s_diff) < 10:
        return float("nan")
    from scipy.stats import mannwhitneyu
    stat = mannwhitneyu(s_same, s_diff, alternative="greater")
    return float(stat.statistic / (len(s_same) * len(s_diff)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scales", default="0.5,1.0,1.5,2.0,3.0")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--controls", default="/home/zizhuo/vcc_data/extracted/context_A.h5ad")
    ap.add_argument("--jitter", default="200.0",
                    help="comma list of jitter shapes for the pds decode sweep")
    ap.add_argument("--pds-mode", choices=("tight", "bayesian"), default="tight")
    ap.add_argument("--pds-amp", default="1.0",
                    help="comma list of delta amplifications for the pds sweep")
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
    print("raw val cosine=%.4f" % (lambda p, t: float(
        ((p * t).sum(1) / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1), 1e-8)).mean()
    ))(raw, eff), flush=True)

    for scale in [float(s) for s in args.scales.split(",")]:
        cal = np.stack([calibrate_like(r, ti, 500, scale) for r, ti in zip(raw, tidx)])
        m = delta_metrics(cal, eff, tidx, args.top_k)
        print("scale=%.2f  fid=%.4f reach=%.4f jac=%.4f nmae=%.4f" % (
            scale, m["fid"], m["reach"], m["jac"], m["nmae"]), flush=True)

    # pds proxy per jitter shape (decode tightness sweep)
    cal = np.stack([calibrate_like(r, ti, 500, 1.0) for r, ti in zip(raw, tidx)])
    import anndata as ad
    controls = ad.read_h5ad(args.controls)
    ctrl_counts = controls.X[:1800]
    for amp in [float(s) for s in args.pds_amp.split(",")]:
        cal_a = cal * amp
        for jt in [float(s) for s in args.jitter.split(",")]:
            auc = pds_proxy(cal_a, ctrl_counts, None, jitter=jt, mode=args.pds_mode)
            print("pds_proxy(mode=%s scale=1.0 amp=%.2f jitter=%.1f)=%.4f"
                  % (args.pds_mode, amp, jt, auc), flush=True)


if __name__ == "__main__":
    main()
