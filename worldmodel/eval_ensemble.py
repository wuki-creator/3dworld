import sys, numpy as np, torch
sys.path.insert(0, '/nfs_beijing_os/zizhuo_vcc/work')
import proxy_score as ps

device = torch.device('cuda')
v = np.load(ps.VAL, allow_pickle=False)

ctrl = v['controls'].astype(np.float32).mean(0)
eff = v['effects'].astype(np.float32)
targets = v['targets'].astype(str)

# reference panel = first member's gene list
model0, genes_ref = ps.load_member(sys.argv[1], device)
ref_lookup = {g: i for i, g in enumerate(genes_ref)}
ok = np.array([t in ref_lookup for t in targets])
eff_ok = eff[ok]
tidx = np.asarray([ref_lookup[t] for t in targets[ok]], dtype=np.int64)

deltas = []
for path in sys.argv[1:]:
    model, genes = ps.load_member(path, device)
    lookup = {g: i for i, g in enumerate(genes)}
    raw = ps.raw_delta(model, ctrl, tidx, device)  # tidx aligned to ref panel
    deltas.append(raw)

avg = np.mean(deltas, axis=0)
cos = float(((avg * eff_ok).sum(1) / np.maximum(np.linalg.norm(avg, axis=1) * np.linalg.norm(eff_ok, axis=1), 1e-8)).mean())
print('ENSEMBLE(n=%d) raw val cosine=%.4f' % (len(deltas), cos))
for scale in (0.5, 1.0):
    cal = np.stack([ps.calibrate_like(r, ti, 500, scale) for r, ti in zip(avg, tidx)])
    m = ps.delta_metrics(cal, eff_ok, tidx)
    print('scale=%.2f  fid=%.4f reach=%.4f jac=%.4f nmae=%.4f' % (scale, m['fid'], m['reach'], m['jac'], m['nmae']))
for path, d in zip(sys.argv[1:], deltas):
    c = float(((d * eff_ok).sum(1) / np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(eff_ok, axis=1), 1e-8)).mean())
    print('  member %s cos=%.4f' % (path.split('/')[-2], c))
