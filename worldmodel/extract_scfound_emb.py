# -*- coding: utf-8 -*-
"""Extract scFoundation gene embeddings for VCC's 18,533 genes.

Path: pos_emb.weight[:19264] (gene token table, 768-d) → PCA-256 → npy.
Output: /nfs_beijing_os/zizhuo_vcc/embeddings/scfound_pca256.npy (18533, 256)
Unmatched genes fall back to hybrid_256 rows (reported).
"""
import numpy as np
import pandas as pd
import torch

SCF = "/nfs_beijing_os/zizhuo_vcc/scfound"
OLD = "/home/zizhuo/cross_cell_vcc_v2/data/vcc_2026"
OUT = "/nfs_beijing_os/zizhuo_vcc/embeddings/scfound_pca256.npy"

# scFoundation gene order (row i of pos_emb = gene i)
scf_genes = pd.read_csv(SCF + "/repo/OS_scRNA_gene_index.19264.tsv",
                        sep="\t")
name_col = "gene_name" if "gene_name" in scf_genes.columns else scf_genes.columns[0]
scf_genes = scf_genes[name_col].astype(str).tolist()
assert len(scf_genes) == 19264, len(scf_genes)
lookup = {g.upper(): i for i, g in enumerate(scf_genes)}

# our genes
our = pd.read_csv(OLD + "/gene_names.csv").iloc[:, 0].astype(str).tolist()
print("our genes:", len(our))

ck = torch.load(SCF + "/cell_model.pt", map_location="cpu", weights_only=False)
pos = ck["pos_emb.weight"][:19264].numpy().astype(np.float32)  # (19264, 768)
print("pos_emb:", pos.shape)

idx = np.array([lookup.get(g.upper(), -1) for g in our])
hit = idx >= 0
print("matched: %d/%d (%.2f%%)" % (hit.sum(), len(our), 100.0 * hit.mean()))

# PCA-256 fit on all scFoundation genes
X = pos - pos.mean(0, keepdims=True)
u, s, vt = np.linalg.svd(X, full_matrices=False)
comp = vt[:256].T.astype(np.float32)          # (768, 256)
emb_all = X @ comp                             # (19264, 256)

emb = np.zeros((len(our), 256), dtype=np.float32)
emb[hit] = emb_all[idx[hit]]
if (~hit).any():  # fallback for unmatched: hybrid_256 (already 256-d)
    hy = np.load("/nfs_beijing_os/zizhuo_vcc/embeddings/vcc_gene_embeddings_hybrid_256.npy")
    assert hy.shape == (len(our), 256)
    emb[~hit] = hy[~hit]
    print("fallback to hybrid for", int((~hit).sum()), "genes")

var = (s[:256] ** 2).sum() / (s ** 2).sum()
print("PCA-256 explained var: %.4f" % var)
np.save(OUT, emb)
print("wrote", OUT, emb.shape)
print("norm mean:", float(np.linalg.norm(emb, axis=1).mean()))
