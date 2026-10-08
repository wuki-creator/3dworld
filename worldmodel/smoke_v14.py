import sys, time, torch
sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
from model_world_h1_v14 import WorldModelH1V14
torch.manual_seed(0)
G, DM, B = 18533, 256, 4
m = WorldModelH1V14(G, d_model=DM).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
ctrl = torch.rand(B, G, device="cuda") * 5.0
pert = torch.randint(0, G, (B,), device="cuda")
t0 = time.time()
for it in range(3):
    opt.zero_grad()
    d = m.predict_delta(ctrl, pert)
    loss = d.pow(2).mean() * 10000.0
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    print("iter", it, "loss", float(loss), flush=True)
params = dict(m.named_parameters())
for name in ["raw_kernel_alpha", "pole.weight", "charge.weight", "response_left.weight", "response_right.weight", "raw_h_max", "gene_bias"]:
    p = params.get(name)
    g = None if p is None or p.grad is None else float(p.grad.abs().max())
    print("GRAD", name, g)
st = (time.time() - t0) / 3
mem = torch.cuda.max_memory_allocated() / 2**30
print(f"SMOKE_OK step_time_s={st:.2f} mem_GB={mem:.1f}")
