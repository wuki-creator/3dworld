# -*- coding: utf-8 -*-
"""补画 velocity 图（latent time / paga）——计算已在上游完成"""
import os
import scanpy as sc
import scvelo as scv
import matplotlib
matplotlib.use("Agg")

VEL = "/nfs_beijing/zizhuo/vcc/results/velocity"
for name in ["pancreas", "dentategyrus"]:
    ad = sc.read(f"{VEL}/{name}_velocity.h5ad")
    try:
        scv.pl.scatter(ad, color="latent_time", color_map="gnuplot",
                       save=f"{VEL}/{name}_latent_time.png", show=False,
                       title=f"{name}: latent time")
    except Exception as e:
        print(name, "latent_time plot fail:", str(e)[:120])
    try:
        scv.pl.paga(ad, save=f"{VEL}/{name}_paga.png", show=False)
    except Exception as e:
        print(name, "paga plot fail:", str(e)[:120])
    print(name, "plots done", flush=True)
print("PLOTS_DONE")
