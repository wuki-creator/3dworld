# -*- coding: utf-8 -*-
"""Convert montage to compact JPEG for slow-link fetch."""
from PIL import Image
import os

src = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs/empiar_sample2/empiar_montage2.png"
dst = "/tmp/empiar_montage2.jpg"
im = Image.open(src).convert("RGB")
w, h = im.size
im = im.resize((int(w * 0.7), int(h * 0.7)))
im.save(dst, quality=62, optimize=True)
print("saved", dst, os.path.getsize(dst))
