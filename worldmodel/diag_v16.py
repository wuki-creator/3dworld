# -*- coding: utf-8 -*-
"""Diagnose v16q: what did the quantum branch actually learn?"""
import sys, json, torch, numpy as np, math
sys.path.insert(0, '/home/zizhuo/maglab_deploy/src')
from model_world_h1_v16 import WorldModelH1V16

ck = torch.load('/nfs_beijing_os/zizhuo_vcc/ckpts/v16q_seed113/magworld_h1_v16q_seed113.pt',
                map_location='cpu', weights_only=False)
m = WorldModelH1V16(**dict(ck['model_config']))
m.load_state_dict(ck['model_state']); m.eval()
t = 2 * torch.tanh(m.qw_time_raw)
gamma = 2 * torch.sigmoid(m.qw_odd_raw) - 1
mix = torch.sigmoid(m.qw_mix).item()
print('t=%.4f  gamma=%.4f  qw_mix_sigmoid=%.4f' % (t.item(), gamma.item(), mix))

E = m.gene_emb.weight
Ac = m.qw_c(E); Ao = m.qw_o(E)
M = (Ao.T @ Ac) / math.sqrt(m.d_z)  # dz x dz, eigenvalues = H's
ev = torch.linalg.eigvalsh((M + M.T) / 2)
print('H sym-part eigenvalue range: [%.4f, %.4f]' % (ev.min().item(), ev.max().item()))

# field vs interaction magnitude on a sample perturbation
v = np.load('/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz')
genes = list(v['genes'].astype(str)); targets = list(v['targets'].astype(str))
lookup = {g: i for i, g in enumerate(genes)}
tidx = torch.tensor([lookup[x] for x in targets[:8]])
idx = m._indices(tidx)
with torch.no_grad():
    qf = m._quantum_field(E, idx)
    src = m.source_code(E[idx])
    rec = m.receiver_code(E) + m.receiver_residual
    inter = src @ rec.T / math.sqrt(m.d_z)
print('quantum_field rms=%.5f max=%.5f | interaction rms=%.5f max=%.5f'
      % (qf.pow(2).mean().sqrt().item(), qf.abs().max().item(),
         inter.pow(2).mean().sqrt().item(), inter.abs().max().item()))
print('quantum_field per-pert norm:', np.round(qf.norm(dim=1).numpy(), 4))
print('interaction per-pert norm:  ', np.round(inter.norm(dim=1).numpy(), 4))

# trainer trajectory comparison
j = json.load(open('/nfs_beijing_os/zizhuo_vcc/ckpts/v16q_seed113/magworld_h1_v16q_seed113.json'))
hist = j.get('history', [])
for ep in (1, 30, 60, 90, 120):
    h = hist[ep - 1]
    print('ep%3d train_loss=%.4f val_cosine=%.4f' % (ep, h['training']['loss'], h['validation']['metrics']['cosine']))
