# -*- coding: utf-8 -*-
"""Quantitative eval: sampled vs GT mito masks."""
import numpy as np

d = np.load("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_runs/empiar_sample1/empiar_sample1.npz")
for k in (0, 4):
    gt = d[f"sample{k}_gt"]
    pr = d[f"sample{k}_pred"]
    thr = float(np.percentile(pr, 99.0))
    pb = pr > thr
    gb = gt > 0.5
    inter = float((pb & gb).sum())
    union = float((pb | gb).sum())
    dice = 2 * inter / max(pb.sum() + gb.sum(), 1)
    iou = inter / max(union, 1)
    corr = float(np.corrcoef(gt.ravel(), pr.ravel())[0, 1])
    print("sample%d: gt_fg=%.4f pred_fg=%.4f IoU=%.4f dice=%.4f corr=%.3f"
          % (k, gb.mean(), pb.mean(), iou, dice, corr), flush=True)
