# -*- coding: utf-8 -*-
"""Explore EMPIAR-10791 1857 segmentation TIFF structure."""
import glob
import os
import numpy as np
import tifffile

BASE = "/local_nvme_data/zizhuo/virtual_life/empiar-10791/1857-obese-climp63/unpacked"
for d in sorted(os.listdir(BASE)):
    sub = os.path.join(BASE, d)
    if not os.path.isdir(sub):
        continue
    files = sorted(glob.glob(os.path.join(sub, "*.tif*")))
    print("DIR:", d, "| files:", len(files))
    for f in files[:2]:
        print("   ", os.path.basename(f))
    if files:
        mid = files[len(files) // 2]
        info = tifffile.imread(mid)
        print("    mid shape", info.shape, "dtype", info.dtype,
              "min", int(info.min()), "max", int(info.max()))
