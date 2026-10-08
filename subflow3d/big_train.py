# -*- coding: utf-8 -*-
"""Train with enlarged 3D velocity field (32 channels, 4 blocks)."""
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
    "--data", "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-p2-v3.npz",
    "--stage", "p2",
    "--target-key", "organelle",
    "--output", "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs/empiar1857_mito_p2_big.pt",
    "--crop-size", "64",
    "--max-samples", "160",
    "--max-steps", "10000",
    "--batch-size", "8",
    "--learning-rate", "2e-4",
    "--device", "cuda",
]
tv.main()
