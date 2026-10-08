# -*- coding: utf-8 -*-
"""Sample EMPIAR mito model v2, eval IoU/DICE vs GT, save montage."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/zizhuo/subflow-matchv1")
from train_volume import VolumeModelConfig, VolumeVelocityField  # noqa: E402

RUNS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs")
DS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets")
OUT = RUNS / "empiar_sample2"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"

src = np.load(DS / "empiar1857-mito-p2.npz")
cond_all = torch.from_numpy(src["condition"].astype(np.float32)).to(DEVICE)
gt_all = src["organelle"].astype(np.float32)
n = cond_all.shape[0]

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


ious, dices = [], []
preds = {}
for idx in (0, 5, 10, 20, 30):
    gt = gt_all[idx, 0]
    pred = euler(cond_all[idx:idx + 1])
    preds[idx] = (cond_all[idx].cpu().numpy(), gt, pred)
    # threshold by matching GT foreground fraction (fair comparison)
    thr = np.percentile(pred, 100 * (1 - (gt > 0.5).mean()))
    pb, gb = pred > thr, gt > 0.5
    inter = (pb & gb).sum()
    iou = inter / max((pb | gb).sum(), 1)
    dice = 2 * inter / max(pb.sum() + gb.sum(), 1)
    corr = float(np.corrcoef(gt.ravel(), pred.ravel())[0, 1])
    ious.append(iou); dices.append(dice)
    print("sample%d: IoU=%.4f dice=%.4f corr=%.3f" % (idx, iou, dice, corr), flush=True)
print("MEAN IoU=%.4f dice=%.4f" % (np.mean(ious), np.mean(dices)), flush=True)

np.savez(OUT / "empiar_sample2.npz",
         **{f"s{k}_gt": v[1] for k, v in preds.items()},
         **{f"s{k}_pred": v[2] for k, v in preds.items()})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n_slices = 8
zvals = np.linspace(4, 59, n_slices).astype(int)
rows = []
for idx, (cond, gt, pred) in preds.items():
    thr = np.percentile(pred, 99.0)
    rows += [(f"s{idx} contour", cond[0]), (f"s{idx} GT", gt),
             (f"s{idx} pred", np.clip(pred, 0, 1)),
             (f"s{idx} thr", (pred > thr).astype(float))]
fig, axes = plt.subplots(len(rows), n_slices, figsize=(2 * n_slices, 2.0 * len(rows)))
for r, (title, vol) in enumerate(rows):
    for k, z in enumerate(zvals):
        ax = axes[r, k]
        ax.imshow(vol[z], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel(title, rotation=0, labelpad=100, va="center", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "empiar_montage2.png", dpi=65)
print("SAVED_PNG", flush=True)
