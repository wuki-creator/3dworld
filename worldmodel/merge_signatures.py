# -*- coding: utf-8 -*-
"""Merge H1 train + val signature npz files into one full-data file."""
import numpy as np

A = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_train_signatures.npz"
B = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz"
OUT = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_trainval_signatures.npz"

da = np.load(A, allow_pickle=False)
db = np.load(B, allow_pickle=False)
assert set(da.files) == set(db.files), (set(da.files), set(db.files))
merged = {}
for k in da.files:
    va, vb = da[k], db[k]
    if k == "genes":
        assert (va == vb).all(), "gene panels differ"
        merged[k] = va
    elif va.shape[1:] == vb.shape[1:]:
        merged[k] = np.concatenate([va, vb], axis=0)
    else:
        raise ValueError("shape mismatch on %s: %s vs %s" % (k, va.shape, vb.shape))
np.savez(OUT, **merged)
d = np.load(OUT, allow_pickle=False)
print("MERGE_OK", {k: d[k].shape for k in ("targets", "replicate_targets", "controls")})
