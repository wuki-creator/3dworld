# -*- coding: utf-8 -*-
"""Basin reproduction verdict: v13 + old geometry vs v13 + new geometry."""
import sys, torch, numpy as np
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

for name, path in [
    ('v13 original      ', '/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed113_np1.pt'),
    ('v13+NEWgeom(ctrl) ', '/nfs_beijing_os/zizhuo_vcc/ckpts/v13_retrain_ctrl/magworld_h1_v13_ctrl_seed113.pt'),
    ('v13+OLDgeom       ', '/nfs_beijing_os/zizhuo_vcc/ckpts/v13geom_seed113/magworld_h1_v13geom_seed113.pt'),
]:
    try:
        m, g = load_member(path, torch.device('cpu'))
    except Exception as e:
        print(name, 'LOAD FAIL:', e, flush=True); continue
    out = np.zeros((len(tidx), len(g)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(tidx), 64):
            e = min(s + 64, len(tidx))
            x = torch.from_numpy(np.tile(ctrl, (e - s, 1)).astype(np.float32))
            out[s:e] = m.predict_delta(x, torch.from_numpy(tidx[s:e])).numpy()
    mr = delta_metrics(out, eff, tidx, top_k=100)
    print('%s cos=%.4f fid=%.4f reach=%.4f jac=%.4f nmae=%.4f'
          % (name, cosine(out, eff), mr['fid'], mr['reach'], mr['jac'], mr['nmae']), flush=True)
