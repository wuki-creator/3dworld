import scanpy as sc
import pandas as pd
ad = sc.read_h5ad("/nfs_beijing/zizhuo/vcc/results/velocity/pancreas_velocity.h5ad")
print(list(ad.var_names[:8]))
net = pd.read_csv(
    "/nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv"
)[["source", "target"]].dropna()
low = set(g.lower() for g in ad.var_names)
tg = set(t.lower() for t in net.target if isinstance(t, str))
print("overlap:", len(low & tg), "of", len(tg))
tf_tg = net[net.source == "RELA"].target
print("RELA targets:", [t for t in tf_tg if isinstance(t, str)][:10])
print("RELA matched:",
      [t for t in tf_tg if isinstance(t, str) and t.lower() in low])
