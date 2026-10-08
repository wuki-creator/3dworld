# -*- coding: utf-8 -*-
"""探测 decoupler 2.x ulm/mlm 的输出结构"""
import scanpy as sc
import decoupler as dc

ad = sc.read_h5ad("/nfs_beijing/zizhuo/vcc/results/core/pbmc3k_processed.h5ad")
net = dc.op.collectri(organism="human")
before = set(ad.obsm.keys())
out = dc.mt.ulm(ad, net, verbose=False)
with open("/tmp/dc8.txt", "w") as f:
    f.write("type=" + repr(type(out)) + "\n")
    f.write("shape=" + repr(getattr(out, "shape", None)) + "\n")
    f.write("new_obsm=" + repr([k for k in ad.obsm.keys() if k not in before]) + "\n")
    f.write("uns_like=" + repr([k for k in ad.uns.keys() if "ulm" in k.lower() or "estimate" in k.lower() or "col" in k.lower()]) + "\n")
    try:
        f.write("cols=" + repr(list(out.columns[:5])) + "\n")
    except Exception as e:
        f.write("nocols " + str(e) + "\n")
    try:
        f.write("varm=" + repr(list(ad.varm.keys())) + "\n")
    except Exception:
        pass
    f.write("DONE\n")
print("PROBE_DONE")
