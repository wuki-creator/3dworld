# -*- coding: utf-8 -*-
"""Raw proxy eval: v13 vs v16q (quantum) head-to-head on h1_val."""
import sys, json, torch, numpy as np
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

def signacc(p, t, k=100):
    return float(np.mean([float((np.sign(p[i][np.argsort(-np.abs(t[i]))[:k]]) == np.sign(t[i][np.argsort(-np.abs(t[i]))[:k]])).mean()) for i in range(len(p))]))

CKPTS = [
    ('v13', '/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed113_np1.pt'),
    ('v16q', '/nfs_beijing_os/zizhuo_vcc/ckpts/v16q_seed113/magworld_h1_v16q_seed113.pt'),
]
for name, path in CKPTS:
    m, g = load_member(path, torch.device('cpu'))
    out = np.zeros((len(tidx), len(g)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(tidx), 64):
            e = min(s + 64, len(tidx))
            x = torch.from_numpy(np.tile(ctrl, (e - s, 1)).astype(np.float32))
            out[s:e] = m.predict_delta(x, torch.from_numpy(tidx[s:e])).numpy()
    mr = delta_metrics(out, eff, tidx, top_k=100)
    print('%s RAW cos=%.4f fid=%.4f reach=%.4f jac=%.4f nmae=%.4f signacc=%.4f'
          % (name, cosine(out, eff), mr['fid'], mr['reach'], mr['jac'], mr['nmae'], signacc(out, eff)), flush=True)
    if name == 'v16q':
        j = json.load(open('/nfs_beijing_os/zizhuo_vcc/ckpts/v16q_seed113/magworld_h1_v16q_seed113.json'))
        print('v16q trainer best_epoch:', j.get('best_epoch'), flush=True)
