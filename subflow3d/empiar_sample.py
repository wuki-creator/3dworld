# -*- coding: utf-8 -*-
"""Sample EMPIAR mito model + montage comparison with GT."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/zizhuo/subflow-matchv1")
from train_volume import VolumeModelConfig, VolumeVelocityField  # noqa: E402

RUNS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs")
DS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets")
OUT = RUNS / "empiar_sample1"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"

src = np.load(DS / "empiar1857-mito-p2.npz")
cond_all = torch.from_numpy(src["condition"].astype(np.float32)).to(DEVICE)
gt_all = src["organelle"].astype(np.float32)

ck = torch.load(RUNS / "empiar1857_mito_p2.pt", map_location=DEVICE)
model = VolumeVelocityField(ck["metadata"]["target_channels"], VolumeModelConfig(**ck["config"])).to(DEVICE)
model.load_state_dict(ck["model"])
model.eval()


@torch.no_grad()
def euler(cond, steps=50):
    x = torch.randn(1, 1, *cond.shape[2:], device=DEVICE)
    dt = 1.0 / steps
    for i in range(1, steps + 1):
        t = torch.full((1,), i * dt, device=DEVICE)
        x = x + dt * model(x, t, cond)
    return x[0, 0].cpu().numpy()


samples = {}
for idx in (0, 4):
    cond = cond_all[idx:idx + 1]
    s = euler(cond)
    samples[idx] = (cond[0].cpu().numpy(), gt_all[idx, 0], s)
    print("SAMPLED", idx, "raw range %.2f..%.2f" % (s.min(), s.max()), flush=True)

np.savez(OUT / "empiar_sample1.npz", **{f"sample{k}_gt": v[1] for k, v in samples.items()},
         **{f"sample{k}_pred": v[2] for k, v in samples.items()})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n = 8
zvals = np.linspace(4, 59, n).astype(int)
fig, axes = plt.subplots(4 * len(samples), n, figsize=(2 * n, 2.1 * 4 * len(samples)))
row = 0
for idx, (cond, gt, pred) in samples.items():
    thr = np.percentile(pred, 99.0)
    for title, vol in [
        (f"s{idx} cell contour", cond[0]),
        (f"s{idx} GT mito", gt),
        (f"s{idx} sampled", np.clip(pred, 0, 1)),
        (f"s{idx} sampled thr>{thr:.2f}", (pred > thr).astype(float)),
    ]:
        for k, z in enumerate(zvals):
            ax = axes[row, k]
            ax.imshow(vol[z], cmap="gray", vmin=0, vmax=1 if vol.max() <= 1.1 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(title, rotation=0, labelpad=110, va="center", fontsize=10)
        row += 1
fig.tight_layout()
fig.savefig(OUT / "empiar_montage.png", dpi=68)
print("SAVED_PNG", flush=True)
