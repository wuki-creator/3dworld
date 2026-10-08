"""Bounded 3D conditional Flow Matching trainer for cropped microscopy volumes.

This is the volume counterpart to :mod:`train`.  It intentionally accepts a
small, explicit NPZ contract so a server job can be limited by crop size,
sample count, and optimizer steps:

* ``condition``: ``[N, 2, Z, Y, X]`` contour and signed-distance channels.
* ``image`` (P1) or a caller-selected ``target``/organellar key (P2):
  ``[N, C, Z, Y, X]`` float arrays.

The trainer does not download data and never silently treats an image channel
as an organelle label.  For P2 the caller must provide ``--target-key`` and
the key must be present in the NPZ.  This keeps the smoke run useful for
checking server CUDA, crop staging, and model plumbing without claiming a
validated organelle model.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class VolumeModelConfig:
    condition_channels: int = 2
    base_channels: int = 16
    blocks: int = 3


class VolumeVelocityField(nn.Module):
    """Small 3D velocity field suitable for bounded server smoke runs."""

    def __init__(self, target_channels: int, config: VolumeModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or VolumeModelConfig()
        self.target_channels = target_channels
        channels = self.config.base_channels
        self.input = nn.Conv3d(target_channels + self.config.condition_channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, channels),
                    nn.SiLU(),
                    nn.Conv3d(channels, channels, 3, padding=1),
                    nn.GroupNorm(4, channels),
                    nn.SiLU(),
                    nn.Conv3d(channels, channels, 3, padding=1),
                )
                for _ in range(self.config.blocks)
            ]
        )
        self.time = nn.Sequential(nn.Linear(2, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.output = nn.Sequential(nn.GroupNorm(4, channels), nn.SiLU(), nn.Conv3d(channels, target_channels, 3, padding=1))

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if state.ndim != 5 or condition.ndim != 5:
            raise ValueError("state and condition must be [B,C,Z,Y,X] tensors")
        if state.shape[0] != condition.shape[0] or state.shape[2:] != condition.shape[2:]:
            raise ValueError("state and condition batch/spatial dimensions must match")
        if state.shape[1] != self.target_channels or condition.shape[1] != self.config.condition_channels:
            raise ValueError("channel count does not match model configuration")
        time_features = self.time(torch.stack((time, time.square()), dim=-1))
        hidden = self.input(torch.cat((state, condition), dim=1))
        for block in self.blocks:
            residual = block(hidden)
            hidden = hidden + residual + time_features[:, :, None, None, None]
        return self.output(hidden)


class CroppedVolumeDataset(Dataset):
    def __init__(self, path: Path, target_key: str, crop_size: int, max_samples: int | None, seed: int, focus_key: str | None) -> None:
        source = np.load(path, allow_pickle=False)
        if "condition" not in source:
            raise ValueError(f"{path} must contain a condition array")
        if target_key not in source:
            available = ", ".join(source.files)
            raise ValueError(f"{path} does not contain target key {target_key!r}; available: {available}")
        condition = np.asarray(source["condition"], dtype=np.float32)
        target = np.asarray(source[target_key], dtype=np.float32)
        # Microscopy channels may be uint8 or higher-bit intensities while
        # binary organelle labels are already in {0, 1}; keep both targets in
        # a numerically stable range for the flow-matching objective.
        if float(np.nanmax(target)) > 1.0:
            scale = max(float(np.nanpercentile(target, 99.5)), 1e-6)
            target = np.clip(target / scale, 0.0, 1.0)
        if condition.ndim != 5 or condition.shape[1] != 2:
            raise ValueError("condition must have shape [N, 2, Z, Y, X]")
        if target.ndim != 5 or target.shape[0] != condition.shape[0] or target.shape[2:] != condition.shape[2:]:
            raise ValueError("target must have shape [N, C, Z, Y, X] paired with condition")
        if crop_size < 8:
            raise ValueError("crop_size must be at least 8 voxels")
        self.condition = condition
        self.target = target
        self.focus_key = focus_key or ("cell_mask" if "cell_mask" in source else "condition")
        if self.focus_key not in source:
            raise ValueError(f"focus key {self.focus_key!r} is not present in the NPZ")
        self.focus = np.asarray(source[self.focus_key], dtype=np.float32)
        if self.focus.ndim != 5 or self.focus.shape[0] != condition.shape[0] or self.focus.shape[2:] != condition.shape[2:]:
            raise ValueError("focus array must be paired with condition")
        self.crop_size = crop_size
        self.indices = np.arange(len(target), dtype=np.int64)
        if max_samples is not None:
            self.indices = self.indices[: max(0, max_samples)]
        self.seed = seed

    def __len__(self) -> int:
        return int(len(self.indices))

    def _crop(self, volume: np.ndarray, index: int, offset: tuple[int, int, int]) -> np.ndarray:
        _, depth, height, width = volume.shape
        starts = tuple(max(0, min(offset[axis], volume.shape[axis + 1] - self.crop_size)) for axis in range(3))
        slices = tuple(slice(starts[axis], starts[axis] + self.crop_size) for axis in range(3))
        result = volume[(slice(None), *slices)]
        padding = [(0, 0)]
        for size in (depth, height, width):
            padding.append((0, max(0, self.crop_size - size)))
        return np.pad(result, padding, mode="constant")

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        index = int(self.indices[item])
        rng = random.Random(self.seed + index)
        shape = self.target.shape[2:]
        foreground = np.argwhere(self.focus[index, 0] > 0)
        if len(foreground):
            center = foreground[rng.randrange(len(foreground))]
            offset = tuple(
                max(0, min(int(center[axis]) - self.crop_size // 2 + rng.randint(-(self.crop_size // 4), self.crop_size // 4), shape[axis] - self.crop_size))
                for axis in range(3)
            )
        else:
            offset = tuple(rng.randrange(max(1, dimension - self.crop_size + 1)) for dimension in shape)
        target = self._crop(self.target[index], index, offset)
        condition = self._crop(self.condition[index], index, offset)
        return torch.from_numpy(target.copy()), torch.from_numpy(condition.copy())


def sample_ot_path(target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    batch = target.shape[0]
    time = torch.rand(batch, device=target.device).clamp_min(1e-4)
    source = torch.randn_like(target)
    t = time.view(batch, 1, 1, 1, 1)
    return (1 - t) * source + t * target, time, target - source


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    requested_device = args.device
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else ("cpu" if requested_device == "auto" else requested_device))
    dataset = CroppedVolumeDataset(Path(args.data), args.target_key, args.crop_size, args.max_samples, args.seed, args.focus_key)
    if not len(dataset):
        raise ValueError("no samples selected")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    target_channels = int(dataset.target.shape[1])
    model = VolumeVelocityField(target_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    losses: list[float] = []
    step = 0
    while step < args.max_steps:
        for target, condition in loader:
            target = target.to(device)
            condition = condition.to(device)
            location, time, target_velocity = sample_ot_path(target)
            loss = torch.nn.functional.mse_loss(model(location, time, condition), target_velocity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            losses.append(float(loss.detach().cpu()))
            if step >= args.max_steps:
                break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": args.stage,
        "data": str(Path(args.data).resolve()),
        "target_key": args.target_key,
        "focus_key": dataset.focus_key,
        "target_channels": target_channels,
        "crop_size": args.crop_size,
        "samples": len(dataset),
        "steps": step,
        "device": str(device),
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
    torch.save({"model": model.state_dict(), "config": asdict(model.config), "metadata": metadata}, output)
    report = {**metadata, "checkpoint": str(output.resolve())}
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="cropped volume NPZ")
    parser.add_argument("--stage", choices=("p0", "p1", "p2"), required=True)
    parser.add_argument("--target-key", default="image", help="P1 defaults to image; P2 must be an explicit organelle target key")
    parser.add_argument("--focus-key", help="foreground array used to center runtime crops (default: cell_mask, then condition)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.stage == "p2" and args.target_key == "image":
        parser.error("P2 requires --target-key pointing to an organelle-labelled target; refusing to use image as a label")
    train(args)


if __name__ == "__main__":
    main()
