# -*- coding: utf-8 -*-
"""Export v13's trained gene table (= lost pre-14:12 hybrid geometry) as npy,
and check s113 vs s227 table agreement."""
import numpy as np, torch

base = '/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat'
out = '/nfs_beijing_os/zizhuo_vcc/embeddings/vcc_gene_embeddings_v13geom_256.npy'

t113 = torch.load(base + '/magworld_h1_v13_full_seed113_np1.pt', map_location='cpu', weights_only=False)['model_state']['gene_emb.weight'].numpy().astype(np.float32)
t227 = torch.load(base + '/magworld_h1_v13_full_seed227_np1.pt', map_location='cpu', weights_only=False)['model_state']['gene_emb.weight'].numpy().astype(np.float32)
d = np.abs(t113 - t227)
cos = (t113 * t227).sum(1) / (np.linalg.norm(t113, axis=1) * np.linalg.norm(t227, axis=1) + 1e-12)
print('s113 vs s227 table: max|diff|=%.6f mean_cos=%.4f' % (d.max(), cos.mean()))

np.save(out, t113)
print('saved', out, t113.shape)

# also sanity: gene order must match the panel
sig = np.load('/nfs_beijing_os/zizhuo_vcc/signatures/h1_trainval_signatures.npz')
ck = torch.load(base + '/magworld_h1_v13_full_seed113_np1.pt', map_location='cpu', weights_only=False)
print('genes order match:', list(ck['genes']) == list(sig['genes'].astype(str)))
