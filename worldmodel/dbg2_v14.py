import sys, torch
sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
from model_world_h1_v14 import WorldModelH1V14
torch.manual_seed(0)
G, DM, B = 18533, 256, 4
m = WorldModelH1V14(G, d_model=DM).cuda()
m._debug = True
ctrl = torch.rand(B, G, device="cuda") * 5.0
pert = torch.randint(0, G, (B,), device="cuda")
d = m.predict_delta(ctrl, pert)
print("DBG", m._dbg)
print("d min/mean/max", float(d.min()), float(d.mean()), float(d.max()))
print("d absmean over genes>1e-3:", float((d.abs() > 1e-3).float().mean()))
print("d top5 abs", float(d.abs().flatten().topk(5).values[-1]))
