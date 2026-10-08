# -*- coding: utf-8 -*-
"""Eval SDF model: sample SDF, zero-level-set -> mask, IoU/DICE vs GT mask."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/zizhuo/subflow-matchv1")
from train_volume import VolumeModelConfig, VolumeVelocityField  # noqa: E402

RUNS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs")
DS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets")
DEVICE = "cuda"

src = np.load(DS / "empiar1857-mito-p2-v3.npz")  # GT binary masks
cond_all = torch.from_numpy(src["condition"].astype(np.float32)).to(DEVICE)
gt_all = src["organelle"].astype(np.float32)

ck = torch.load(RUNS / "empiar1857_mito_sdf_big.pt", map_location=DEVICE)
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


indices = [0, 10, 25, 50, 75, 100, 125, 135]
ious, dices = [], []
for idx in indices:
    gt = gt_all[idx, 0]
    sdf = euler(cond_all[idx:idx + 1])
    pb, gb = sdf > 0, gt > 0.5
    inter = (pb & gb).sum()
    iou = inter / max((pb | gb).sum(), 1)
    dice = 2 * inter / max(pb.sum() + gb.sum(), 1)
    corr = float(np.corrcoef(gt.ravel(), sdf.ravel())[0, 1])
    ious.append(iou); dices.append(dice)
    print("sample%d: IoU=%.4f dice=%.4f corr=%.3f fg_pred=%.4f fg_gt=%.4f"
          % (idx, iou, dice, corr, pb.mean(), gb.mean()), flush=True)
print("SDF_MODEL MEAN IoU=%.4f dice=%.4f" % (np.mean(ious), np.mean(dices)), flush=True)
