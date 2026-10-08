# -*- coding: utf-8 -*-
"""Train big 3D velocity field on SDF targets (v4)."""
import sys
from dataclasses import dataclass

import train_volume as tv


@dataclass(frozen=True)
class BigConfig:
    condition_channels: int = 2
    base_channels: int = 32
    blocks: int = 4


tv.VolumeModelConfig = BigConfig
sys.argv = [
    "train_volume.py",
    "--data", "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-sdf-v4.npz",
    "--stage", "p2",
    "--target-key", "organelle",
    "--output", "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs/empiar1857_mito_sdf_big.pt",
    "--crop-size", "64",
    "--max-samples", "136",
    "--max-steps", "10000",
    "--batch-size", "8",
    "--learning-rate", "2e-4",
    "--device", "cuda",
]
tv.main()
