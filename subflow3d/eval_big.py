# -*- coding: utf-8 -*-
"""Eval big model vs v2 model on the same v3 samples."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/zizhuo/subflow-matchv1")
from train_volume import VolumeModelConfig, VolumeVelocityField  # noqa: E402

RUNS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs")
DS = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets")
DEVICE = "cuda"

src = np.load(DS / "empiar1857-mito-p2-v3.npz")
cond_all = torch.from_numpy(src["condition"].astype(np.float32)).to(DEVICE)
gt_all = src["organelle"].astype(np.float32)


def load(name):
    ck = torch.load(RUNS / name, map_location=DEVICE)
    cfg = VolumeModelConfig(**ck["config"])
    m = VolumeVelocityField(ck["metadata"]["target_channels"], cfg).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def euler(model, cond, steps=50):
    x = torch.randn(1, 1, *cond.shape[2:], device=DEVICE)
    dt = 1.0 / steps
    for i in range(1, steps + 1):
        t = torch.full((1,), i * dt, device=DEVICE)
        x = x + dt * model(x, t, cond)
    return x[0, 0].cpu().numpy()


models = {}
for name in ("empiar1857_mito_p2.pt", "empiar1857_mito_p2_big.pt"):
    try:
        models[name] = load(name)
    except Exception as e:
        print(name, "LOAD_FAIL", str(e)[:80], flush=True)

indices = [0, 10, 25, 50, 75, 100, 125, min(150, cond_all.shape[0] - 1)]
for mname, model in models.items():
    ious, dices = [], []
    for idx in indices:
        if idx >= cond_all.shape[0]:
            continue
        gt = gt_all[idx, 0]
        pred = euler(model, cond_all[idx:idx + 1])
        thr = np.percentile(pred, 100 * (1 - (gt > 0.5).mean()))
        pb, gb = pred > thr, gt > 0.5
        inter = (pb & gb).sum()
        ious.append(inter / max((pb | gb).sum(), 1))
        dices.append(2 * inter / max(pb.sum() + gb.sum(), 1))
    print("%s: meanIoU=%.4f meanDice=%.4f (n=%d)" % (mname, np.mean(ious), np.mean(dices), len(ious)), flush=True)
