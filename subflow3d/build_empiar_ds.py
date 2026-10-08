# -*- coding: utf-8 -*-
"""Build 3D flow-matching dataset from EMPIAR-10791 1857 segmentations.

condition = [contour, sdf] of cell mask; target = mitochondria binary mask.
Crops: 256x256x64 windows around the largest cell in each z-window,
split into 4x 64^3 sub-crops (2x2 in y,x).
"""
import glob
import os

import numpy as np
import tifffile
from skimage.measure import label, regionprops
from scipy import ndimage

BASE = "/local_nvme_data/zizhuo/virtual_life/empiar-10791/1857-obese-climp63/unpacked"
CELL = os.path.join(BASE, "HSPH_obob_Climp63OE_1857 cell segmentation")
MITO = os.path.join(BASE, "1857 Mito")
OUT = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-p2.npz"

cell_files = sorted(glob.glob(os.path.join(CELL, "*.tif*")))
mito_files = sorted(glob.glob(os.path.join(MITO, "*.tif*")))
assert len(cell_files) == len(mito_files) == 3629
print("file lists ok", flush=True)

Z_WINDOWS = [(1800, 64), (900, 64), (2700, 64)]
WIN = 256
SUB = 64

conditions = []
targets = []


def contour_sdf(mask):
    """mask: bool 3D -> [2,Z,Y,X] float32 contour+sdf."""
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


for z0, nz in Z_WINDOWS:
    zmid = z0 + nz // 2
    cseg = tifffile.imread(cell_files[zmid])
    lab = label(cseg > 0)
    props = sorted(regionprops(lab), key=lambda r: -r.area)
    if not props:
        print("no cell at zmid", zmid, flush=True)
        continue
    cy, cx = [int(v) for v in props[0].centroid]
    y0 = min(max(cy - WIN // 2, 0), cseg.shape[0] - WIN)
    x0 = min(max(cx - WIN // 2, 0), cseg.shape[1] - WIN)
    print(f"zwin {z0}: cell area={props[0].area} center=({cy},{cx}) window=({y0},{x0})", flush=True)

    cell_blk = np.empty((nz, WIN, WIN), dtype=np.uint8)
    mito_blk = np.empty((nz, WIN, WIN), dtype=np.uint8)
    for i in range(nz):
        cs = tifffile.imread(cell_files[z0 + i])
        cell_blk[i] = cs[y0:y0 + WIN, x0:x0 + WIN]
        ms = tifffile.imread(mito_files[z0 + i])
        mito_blk[i] = ms[y0:y0 + WIN, x0:x0 + WIN]
    cell_mask = cell_blk > 0
    mito_mask = (mito_blk > 0).astype(np.float32)

    for dy in (0, WIN - SUB):
        for dx in (0, WIN - SUB):
            cm = cell_mask[:, dy:dy + SUB, dx:dx + SUB]
            if cm.mean() < 0.05:
                continue
            conditions.append(contour_sdf(cm))
            targets.append(mito_mask[:, dy:dy + SUB, dx:dx + SUB][None])

cond = np.stack(conditions)
targ = np.stack(targets)
print("dataset:", cond.shape, targ.shape, flush=True)
print("target foreground frac:", float(targ.mean()), flush=True)
np.savez(OUT, condition=cond.astype(np.float32), organelle=targ.astype(np.float32))
print("SAVED", OUT, os.path.getsize(OUT), flush=True)

# preview montage of first sample
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
n = 8
zvals = np.linspace(4, SUB - 5, n).astype(int)
fig, axes = plt.subplots(4, n, figsize=(2 * n, 8))
for c, (title, vol) in enumerate([
    ("cell contour", cond[0, 0]), ("cell sdf", cond[0, 1]),
    ("mito target", targ[0, 0]),
    ("cell mask", (cond[0, 0] > 0).astype(float)),
]):
    for k, z in enumerate(zvals):
        axes[c, k].imshow(vol[z], cmap="gray")
        axes[c, k].set_xticks([]); axes[c, k].set_yticks([])
        if k == 0:
            axes[c, 0].set_ylabel(title, rotation=0, labelpad=90, va="center")
fig.tight_layout()
fig.savefig("/tmp/empiar_preview.png", dpi=70)
print("SAVED_PREVIEW", flush=True)
