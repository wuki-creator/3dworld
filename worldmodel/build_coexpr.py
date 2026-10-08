# -*- coding: utf-8 -*-
"""Build coexpression kNN prior graph from coexpression_128.npy (v25 npz format)."""
import numpy as np

E = "/nfs_beijing_os/zizhuo_vcc/embeddings"
x = np.load(E + "/coexpression_128.npy").astype(np.float32)
G = x.shape[0]
k = 50
sim = x @ x.T
np.fill_diagonal(sim, -1e9)
nb = np.argpartition(-sim, k, axis=1)[:, :k]
src = np.repeat(np.arange(G), k)
dst = nb.reshape(-1)
w = np.clip(sim[src, dst], 0, None).astype(np.float32)
np.savez(E + "/coexpr_graph.npz",
         edge_index=np.stack([src, dst]).astype(np.int64),
         edge_sign=np.ones(len(src), dtype=np.float32),
         edge_w=w)
print("coexpr_graph.npz:", len(src), "edges, w mean", float(w.mean()))
