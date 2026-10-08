# -*- coding: utf-8 -*-
"""Build EMPIAR mito 3D dataset v2: mito-aware window selection.

For each z-window: load 64-slice mito + cell-seg stacks, project mito-inside-cell
onto xy, greedily pick top non-overlapping 256x256 windows, then cut 4 sub-crops
64^3 per window. Keep crops with mito fg >= 0.5% and cell occupancy >= 5%.
"""
import glob
import os

import numpy as np
import tifffile
from skimage.measure import block_reduce
from scipy import ndimage

BASE = "/local_nvme_data/zizhuo/virtual_life/empiar-10791/1857-obese-climp63/unpacked"
CELL = os.path.join(BASE, "HSPH_obob_Climp63OE_1857 cell segmentation")
MITO = os.path.join(BASE, "1857 Mito")
OUT = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-p2.npz"

cell_files = sorted(glob.glob(os.path.join(CELL, "*.tif*")))
mito_files = sorted(glob.glob(os.path.join(MITO, "*.tif*")))

Z_WINDOWS = [(1800, 64), (900, 64), (2700, 64)]
WIN = 256
SUB = 64


def contour_sdf(mask):
    mask = mask.astype(bool)
    er = ndimage.binary_erosion(mask, iterations=1)
    contour = (mask & ~er).astype(np.float32)
    dt_in = ndimage.distance_transform_edt(mask)
    dt_out = ndimage.distance_transform_edt(~mask)
    sdf = (dt_in - dt_out).astype(np.float32)
    m = np.abs(sdf).max()
    if m > 0:
        sdf /= m
    return np.stack([contour, sdf]).astype(np.float32)


conditions, targets = [], []
for z0, nz in Z_WINDOWS:
    cell_blk = np.empty((nz, 9650, 9700), dtype=bool)
    mito_blk = np.empty((nz, 9650, 9700), dtype=bool)
    for i in range(nz):
        cell_blk[i] = tifffile.imread(cell_files[z0 + i]) > 0
        mito_blk[i] = tifffile.imread(mito_files[z0 + i]) > 0
        if i % 16 == 0:
            print(f"zwin {z0}: slice {i}/{nz}", flush=True)
    proj = (mito_blk & cell_blk).sum(axis=0).astype(np.float32)
    grid = block_reduce(proj, (WIN, WIN), func=np.sum)  # [~37, ~37]
    flat = []
    for gy in range(grid.shape[0]):
        for gx in range(grid.shape[1]):
            flat.append((grid[gy, gx], gy * WIN, gx * WIN))
    flat.sort(key=lambda t: -t[0])
    chosen = []
    for score, y, x in flat:
        if score <= 0:
            break
        if all(abs(y - cy) >= WIN or abs(x - cx) >= WIN for _, cy, cx in chosen):
            chosen.append((score, y, x))
        if len(chosen) >= 3:
            break
    print(f"zwin {z0}: windows {[(int(s), y, x) for s, y, x in chosen]}", flush=True)
    for score, y, x in chosen:
        cm = cell_blk[:, y:y + WIN, x:x + WIN]
        mm = mito_blk[:, y:y + WIN, x:x + WIN]
        for dy in (0, WIN - SUB):
            for dx in (0, WIN - SUB):
                csub = cm[:, dy:dy + SUB, dx:dx + SUB]
                msub = mm[:, dy:dy + SUB, dx:dx + SUB].astype(np.float32)
                if csub.mean() < 0.05 or msub.mean() < 0.005:
                    continue
                conditions.append(contour_sdf(csub))
                targets.append(msub[None])
    del cell_blk, mito_blk

cond = np.stack(conditions)
targ = np.stack(targets)
print("dataset:", cond.shape, targ.shape, flush=True)
print("fg per sample:", ["%.4f" % t.mean() for t in targ], flush=True)
np.savez(OUT, condition=cond.astype(np.float32), organelle=targ.astype(np.float32))
print("SAVED", OUT, os.path.getsize(OUT), flush=True)
