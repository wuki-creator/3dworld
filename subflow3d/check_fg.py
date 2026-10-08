# -*- coding: utf-8 -*-
"""Per-sample foreground stats of empiar dataset + resample eval on best sample."""
import numpy as np

d = np.load("/local_nvme_data/zizhuo/virtual_life/subflow_matchv1_datasets/empiar1857-mito-p2.npz")
t = d["organelle"]
for i in range(t.shape[0]):
    print("train_sample %d fg=%.5f" % (i, t[i].mean()), flush=True)
