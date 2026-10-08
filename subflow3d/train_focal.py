# -*- coding: utf-8 -*-
"""Focal-weighted flow-matching training on binary mito masks.

Same architecture/data contract as train_volume.py, but the MSE on the
velocity target is weighted: foreground voxels get (1 + W_FG) x weight,
normalized per batch. Counters background-gradient dominance (96% bg).
"""
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/zizhuo/subflow-matchv1")
from train_volume import (  # noqa: E402
    CroppedVolumeDataset,
    VolumeVelocityField,
    sample_ot_path,
)

DATA = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-p2-v3.npz"
OUT = Path("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs/empiar1857_mito_focal.pt")
STEPS = 10000
BATCH = 8
LR = 2e-4
W_FG = 9.0
DEVICE = "cuda"


@dataclass(frozen=True)
class BigConfig:
    condition_channels: int = 2
    base_channels: int = 32
    blocks: int = 4


def main():
    dataset = CroppedVolumeDataset(Path(DATA), "organelle", 64, None, 20261001, None)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=False, num_workers=0)
    model = VolumeVelocityField(int(dataset.target.shape[1]), BigConfig()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    losses = []
    step = 0
    t0 = time.time()
    while step < STEPS:
        for target, condition in loader:
            target = target.to(DEVICE)
            condition = condition.to(DEVICE)
            location, time_t, tv = sample_ot_path(target)
            pred = model(location, time_t, condition)
            per_voxel = (pred - tv) ** 2
            w = 1.0 + W_FG * target
            w = w / w.mean()
            loss = (per_voxel * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(float(loss.detach().cpu()))
            if step % 500 == 0:
                print("step %d/%d loss=%.4f elapsed=%.0fs" % (step, STEPS, losses[-1], time.time() - t0), flush=True)
            if step >= STEPS:
                break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": "p2",
        "data": str(Path(DATA).resolve()),
        "target_key": "organelle",
        "focus_key": dataset.focus_key,
        "target_channels": int(dataset.target.shape[1]),
        "crop_size": 64,
        "samples": len(dataset),
        "steps": step,
        "device": DEVICE,
        "loss": "focal_mse_wfg=%.1f" % W_FG,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
    torch.save({"model": model.state_dict(), "config": asdict(BigConfig()), "metadata": metadata}, OUT)
    report = {**metadata, "checkpoint": str(OUT.resolve())}
    OUT.with_suffix(OUT.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
