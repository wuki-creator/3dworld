# -*- coding: utf-8 -*-
"""Cascade sampling: P0/P1/P2 3D flow-matching models conditioned on a held-out test cell.

Saves generated volumes (npz) and a slice-montage PNG for visual check.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/zizhuo/subflow-matchv1")
from train_volume import VolumeModelConfig, VolumeVelocityField  # noqa: E402

RUNS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs")
DS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets")
OUT = RUNS / "cascade_sample1"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
CROP = 64
STEPS = 50


def load_model(ckpt_name):
    ck = torch.load(RUNS / ckpt_name, map_location=DEVICE)
    cfg = VolumeModelConfig(**ck["config"])
    model = VolumeVelocityField(ck["metadata"]["target_channels"], cfg).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def euler_sample(model, cond, steps=STEPS):
    x = torch.randn(1, model.target_channels, *cond.shape[2:], device=DEVICE)
    dt = 1.0 / steps
    for i in range(1, steps + 1):
        t = torch.full((1,), i * dt, device=DEVICE)
        x = x + dt * model(x, t, cond)
    return x[0].cpu().numpy()


def crop_around(a, center, rng, jitter=8):
    off = [
        int(max(0, min(int(center[ax]) - CROP // 2 + rng.randint(-jitter, jitter + 1),
                       a.shape[ax + 1] - CROP)))
        for ax in range(3)
    ]
    sl = tuple(slice(off[ax], off[ax] + CROP) for ax in range(3))
    r = a[(slice(None),) + sl]
    pad = [(0, 0)] + [(0, max(0, CROP - s)) for s in a.shape[1:]]
    return np.pad(r, pad, mode="constant")


def main():
    src = np.load(DS / "allen-multi-organelle-p2.npz")
    idx = src["condition"].shape[0] - 1  # last record = test split
    cond_full = src["condition"][idx].astype(np.float32)
    fg = np.argwhere(cond_full[0] > 0)
    center = fg[len(fg) // 2]
    rng = np.random.RandomState(7)
    cond_np = crop_around(cond_full, center, rng)[None]  # [1,2,64,64,64]
    cond = torch.from_numpy(cond_np).to(DEVICE)

    results = {
        "condition": cond_np[0],
        "gt_organelle": crop_around(src["organelle"][idx].astype(np.float32), center, rng),
        "gt_image": crop_around(src["image"][idx].astype(np.float32), center, rng),
    }
    for name, ck in [
        ("p2_organelle", "p2_multi_organelle.pt"),
        ("p1_image", "p1_tomm20.pt"),
        ("p0_image", "p0_ctc_a549.pt"),
    ]:
        model = load_model(ck)
        results[name] = euler_sample(model, cond)
        del model
        torch.cuda.empty_cache()
        print("SAMPLED", name, flush=True)

    np.savez(OUT / "cascade_sample1.npz", **results)
    print("SAVED_NPZ", OUT / "cascade_sample1.npz", flush=True)

    # ---- montage PNG ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("condition contour", results["condition"][0], None),
        ("condition sdf", results["condition"][1], None),
        ("GT organelle", results["gt_organelle"][0], (0, 1)),
        ("P2 sampled organelle", results["p2_organelle"][0], None),
        ("GT image", results["gt_image"][0], None),
        ("P1 sampled image (TOMM20)", results["p1_image"][0], None),
        ("P0 sampled image (CTC)", results["p0_image"][0], None),
    ]
    n_slices = 8
    z_vals = np.linspace(8, CROP - 9, n_slices).astype(int)
    fig, axes = plt.subplots(len(rows), n_slices, figsize=(2.0 * n_slices, 2.2 * len(rows)))
    for r, (title, vol, vr) in enumerate(rows):
        for c, z in enumerate(z_vals):
            ax = axes[r, c]
            ax.imshow(vol[z], cmap="gray", vmin=vr[0] if vr else None, vmax=vr[1] if vr else None)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, rotation=0, labelpad=140, va="center", fontsize=11)
            if r == 0:
                ax.set_title(f"z={z}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "cascade_montage.png", dpi=70)
    print("SAVED_PNG", OUT / "cascade_montage.png", flush=True)


if __name__ == "__main__":
    main()
