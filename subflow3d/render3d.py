# -*- coding: utf-8 -*-
"""Render 3D surfaces: translucent cell shell (condition contour) + P2 sampled organelle."""
import numpy as np
from skimage.measure import marching_cubes

D = "/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs/cascade_sample1"
d = np.load(f"{D}/cascade_sample1.npz")
org = d["p2_organelle"][0].astype(np.float32)   # [64,64,64]
cell = d["condition"][0].astype(np.float32)      # contour channel

print("org stats: min=%.3f max=%.3f mean=%.4f p95=%.3f p98=%.3f p99=%.3f"
      % (org.min(), org.max(), org.mean(), *np.percentile(org, [95, 98, 99])), flush=True)

thr = float(np.percentile(org, 97))
print("organelle iso level:", thr, flush=True)

verts_o, faces_o, _, _ = marching_cubes(org, level=thr)
verts_c, faces_c, _, _ = marching_cubes(cell, level=0.5)
print("mesh: organelle", verts_o.shape[0], "verts", faces_o.shape[0], "faces; cell",
      verts_c.shape[0], "verts", faces_c.shape[0], "faces", flush=True)

import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

views = [(25, 45), (25, 135), (65, 225), (10, 315)]
fig = plt.figure(figsize=(16, 16))
for i, (elev, azim) in enumerate(views, 1):
    ax = fig.add_subplot(2, 2, i, projection="3d")
    cell_mesh = Poly3DCollection(verts_c[faces_c], alpha=0.10, facecolor="lightgray",
                                 edgecolor="none")
    org_mesh = Poly3DCollection(verts_o[faces_o], alpha=0.85, facecolor="limegreen",
                                edgecolor="darkgreen", linewidths=0.05)
    ax.add_collection3d(org_mesh)
    ax.add_collection3d(cell_mesh)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"elev={elev} azim={azim}", fontsize=12)
    ax.set_axis_off()
fig.suptitle("3D cell (translucent shell) with P2-sampled organelles (green)", fontsize=15)
fig.tight_layout()
fig.savefig(f"{D}/render3d.png", dpi=45)
print("SAVED", f"{D}/render3d.png", flush=True)
