# -*- coding: utf-8 -*-
"""Two-basin diagnosis:
1. Is original v13 seed227 also in the sharp basin? (if yes: environment changed)
2. Weight statistics: sharp (v13 seed113) vs diffuse (v17c) — what differs?
"""
import sys, torch, numpy as np
sys.path.insert(0, '/home/zizhuo/maglab_deploy/src')
sys.path.insert(0, '/nfs_beijing_os/zizhuo_vcc/work')
from proxy_score import delta_metrics, load_member

v = np.load('/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz')
targets = list(v['targets'].astype(str)); genes = list(v['genes'].astype(str))
lookup = {g: i for i, g in enumerate(genes)}
tidx = np.asarray([lookup[t] for t in targets])
eff = v['effects'].astype(np.float32)
ctrl = v['controls'].astype(np.float32).mean(0)

def cosine(p, t):
    return float(((p * t).sum(1) / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1), 1e-8)).mean())

m, g = load_member('/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed227_np1.pt', torch.device('cpu'))
out = np.zeros((len(tidx), len(g)), dtype=np.float32)
with torch.no_grad():
    for s in range(0, len(tidx), 64):
        e = min(s + 64, len(tidx))
        x = torch.from_numpy(np.tile(ctrl, (e - s, 1)).astype(np.float32))
        out[s:e] = m.predict_delta(x, torch.from_numpy(tidx[s:e])).numpy()
mr = delta_metrics(out, eff, tidx, top_k=100)
print('v13 seed227 RAW cos=%.4f fid=%.4f reach=%.4f nmae=%.4f' % (cosine(out, eff), mr['fid'], mr['reach'], mr['nmae']), flush=True)

# ---- weight stats: sharp vs diffuse ----
def stats(path, mod, cls, tag):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model_state']
    gs = torch.nn.functional.softplus(sd['gene_scale'])
    cs = sd['context_strength']
    se = torch.nn.functional.softplus(sd['raw_self_effect'])
    zd = sd.get('zdelta_scale')
    print('%s: softplus(gene_scale) mean=%.4f std=%.4f | context_strength=%.4f | self_effect=%.4f | zdelta_scale=%s'
          % (tag, gs.mean().item(), gs.std().item(), cs.item(), se.item(),
             ('%.4f' % torch.nn.functional.softplus(zd).item()) if zd is not None else 'n/a'), flush=True)

stats('/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed113_np1.pt', None, None, 'sharp v13s113')
stats('/nfs_beijing_os/zizhuo_vcc/ckpts/v17c_seed113/magworld_h1_v17c_seed113.pt', None, None, 'diffuse v17c')
# also the magnetic force magnitude diagnostic from training logs
import json
for tag, p in [('sharp v13s113', '/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/seed113_meta.json'),
               ('diffuse v17c', '/nfs_beijing_os/zizhuo_vcc/ckpts/v17c_seed113/magworld_h1_v17c_seed113.json')]:
    j = json.load(open(p))
    h = j.get('history', [])
    if h:
        last = h[-1].get('magnetic', {})
        print(tag, 'last-epoch magnetic:', {k: round(float(x), 5) for k, x in last.items() if isinstance(x, (int, float))}, flush=True)
