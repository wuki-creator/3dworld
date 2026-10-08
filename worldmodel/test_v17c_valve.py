# -*- coding: utf-8 -*-
"""Isolate the collapse cause: v17c ckpt with fluid valve forced CLOSED.
If cos jumps to ~0.39 -> the trained fluid params destroy the output.
If stays ~0.17 -> the shared weights were trained into the diffuse basin."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/zizhuo/maglab_deploy/src')
sys.path.insert(0, '/nfs_beijing_os/zizhuo_vcc/work')
from proxy_score import delta_metrics
from model_world_h1_v17c import WorldModelH1V17C

ck = torch.load('/nfs_beijing_os/zizhuo_vcc/ckpts/v17c_seed113/magworld_h1_v17c_seed113.pt',
                map_location='cpu', weights_only=False)
m = WorldModelH1V17C(**dict(ck['model_config']))
m.load_state_dict(ck['model_state']); m.eval()

v = np.load('/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz')
targets = list(v['targets'].astype(str)); genes = list(v['genes'].astype(str))
lookup = {g: i for i, g in enumerate(genes)}
tidx = np.asarray([lookup[t] for t in targets])
eff = v['effects'].astype(np.float32)
ctrl = v['controls'].astype(np.float32).mean(0)

def cosine(p, t):
    return float(((p * t).sum(1) / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1), 1e-8)).mean())

def run(tag):
    out = np.zeros((len(tidx), len(genes)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(tidx), 64):
            e = min(s + 64, len(tidx))
            x = torch.from_numpy(np.tile(ctrl, (e - s, 1)).astype(np.float32))
            out[s:e] = m.predict_delta(x, torch.from_numpy(tidx[s:e])).numpy()
    mr = delta_metrics(out, eff, tidx, top_k=100)
    print('%s cos=%.4f fid=%.4f reach=%.4f nmae=%.4f' % (tag, cosine(out, eff), mr['fid'], mr['reach'], mr['nmae']), flush=True)

with torch.no_grad():
    m.fl_mix.fill_(-20.0)   # fluid weight ~ 2e-9: forward becomes exactly v13 math
run('v17c fluid=CLOSED')
