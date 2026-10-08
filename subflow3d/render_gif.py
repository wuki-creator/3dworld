# -*- coding: utf-8 -*-
"""Render rotating GIF of 3D cell shell + P2-sampled organelles."""
import io
import numpy as np
from skimage.measure import marching_cubes

D = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs/cascade_sample1"
d = np.load(f"{D}/cascade_sample1.npz")
org = d["p2_organelle"][0].astype(np.float32)
cell = d["condition"][0].astype(np.float32)

thr = float(np.percentile(org, 97))
verts_o, faces_o, _, _ = marching_cubes(org, level=thr)
verts_c, faces_c, _, _ = marching_cubes(cell, level=0.5)
print("meshes ready", flush=True)

import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from PIL import Image

N = 16
frames = []
for k in range(N):
    fig = plt.figure(figsize=(3.6, 3.6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(verts_c[faces_c], alpha=1.0,
                                         facecolor="#e0e0e0", edgecolor="none"))
    ax.add_collection3d(Poly3DCollection(verts_o[faces_o], alpha=1.0,
                                         facecolor="#22aa44", edgecolor="#0d5c22", linewidths=0.02))
    ax.view_init(elev=22, azim=k * (360.0 / N))
    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB").quantize(colors=16))
    print("frame", k + 1, "/", N, flush=True)

out = f"{D}/render3d_rotate.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], optimize=True,
               duration=130, loop=0)
import os
print("SAVED", out, os.path.getsize(out), flush=True)
