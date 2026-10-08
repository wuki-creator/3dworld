# -*- coding: utf-8 -*-
"""Build EMPIAR mito 3D dataset v3: scaled-up (9 z-windows, top-4 mito windows each)."""
import glob
import os

import numpy as np
import tifffile
from skimage.measure import block_reduce
from scipy import ndimage

BASE = "/local_nvme_data/zizhuo/virtual_life/empiar-10791/1857-obese-climp63/unpacked"
CELL = os.path.join(BASE, "HSPH_obob_Climp63OE_1857 cell segmentation")
MITO = os.path.join(BASE, "1857 Mito")
OUT = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-p2-v3.npz"

cell_files = sorted(glob.glob(os.path.join(CELL, "*.tif*")))
mito_files = sorted(glob.glob(os.path.join(MITO, "*.tif*")))

Z_WINDOWS = [(300, 64), (640, 64), (980, 64), (1320, 64), (1800, 64),
             (2200, 64), (2600, 64), (3000, 64), (3350, 64)]
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
        if i % 32 == 0:
            print(f"zwin {z0}: slice {i}/{nz}", flush=True)
    proj = (mito_blk & cell_blk).sum(axis=0).astype(np.float32)
    grid = block_reduce(proj, (WIN, WIN), func=np.sum)
    flat = []
    for gy in range(grid.shape[0]):
        for gx in range(grid.shape[1]):
            flat.append((grid[gy, gx], gy * WIN, gx * WIN))
    flat.sort(key=lambda t: -t[0])
    chosen = []
    for score, y, x in flat:
        if score <= 0:
            break
        y = min(y, 9650 - WIN)
        x = min(x, 9700 - WIN)
        if all(abs(y - cy) >= WIN or abs(x - cx) >= WIN for _, cy, cx in chosen):
            chosen.append((score, y, x))
        if len(chosen) >= 4:
            break
    kept = 0
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
                kept += 1
    print(f"zwin {z0}: kept {kept} crops", flush=True)
    del cell_blk, mito_blk

cond = np.stack(conditions)
targ = np.stack(targets)
print("dataset:", cond.shape, targ.shape, flush=True)
fgs = [float(t.mean()) for t in targ]
print("fg min/mean/max: %.4f %.4f %.4f" % (min(fgs), sum(fgs) / len(fgs), max(fgs)), flush=True)
np.savez(OUT, condition=cond.astype(np.float32), organelle=targ.astype(np.float32))
print("SAVED", OUT, os.path.getsize(OUT), flush=True)
