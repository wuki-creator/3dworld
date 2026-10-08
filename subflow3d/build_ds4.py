# -*- coding: utf-8 -*-
"""Convert v3 mito masks to signed-distance-field targets (v4)."""
import numpy as np
from scipy import ndimage

SRC = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-p2-v3.npz"
DST = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-sdf-v4.npz"

d = np.load(SRC)
cond = d["condition"].astype(np.float32)
masks = d["organelle"].astype(np.float32)
n = masks.shape[0]
sdf = np.empty_like(masks)
for i in range(n):
    m = masks[i, 0] > 0.5
    dt_in = ndimage.distance_transform_edt(m)
    dt_out = ndimage.distance_transform_edt(~m)
    s = (dt_in - dt_out).astype(np.float32)
    mx = np.abs(s).max()
    if mx > 0:
        s /= mx
    sdf[i, 0] = s
    if i % 20 == 0:
        print("sdf", i, "/", n, flush=True)
np.savez(DST, condition=cond, organelle=sdf)
print("SAVED", DST, flush=True)
