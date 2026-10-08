import sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
from model_world_h1_v14 import WorldModelH1V14
torch.manual_seed(421)
G, DM, B = 18533, 256, 8
cfg = dict(n_genes=G, d_model=DM, d_z=64, d_hidden=192, d_dir=32, max_delta=1.5, context_strength=0.25, shared_bias_mode="gated", shared_bias_initial_scale=0.1, normalize_context=True, graph_k=50, n_layers=3, response_rank=64, use_dipole=True, use_langevin=True, langevin_init=4.0, kernel_powers=4)
m = WorldModelH1V14(**cfg).to("cuda")
bad = [(n, tuple(p.shape), str(p.device)) for n, p in m.named_parameters() if p.device.type != "cuda"]
badb = [(n, str(p.device)) for n, p in m.named_buffers() if p.device.type != "cuda"]
print("CPU_PARAMS", bad)
print("CPU_BUFFERS", badb)
from train_magworld_h1_v14 import gaussian_mmd, sign_bce
opt = torch.optim.AdamW(m.parameters(), lr=8e-4, weight_decay=2e-4)
ctrl = torch.rand(B, G, device="cuda") * 5.0
yb = torch.randn(B, G, device="cuda") * 0.01
pert = torch.randint(0, G, (B,), device="cuda")
mask = torch.rand(B, G, device="cuda") > 0.9
pred = m.predict_delta(ctrl, pert)
l1 = (pred - yb).abs().mean(1).mean()
mmd = gaussian_mmd(pred, yb)
bce = sign_bce(pred, yb, mask, 0.05)
alpha_init = m.kernel_alpha_init.to("cuda")
kernel_prior = ((m.kernel_alpha().to("cuda") - alpha_init) ** 2).sum()
shared_bias = m.effective_shared_bias_scale() * m.gene_bias.abs().mean()
loss = 1.0 * l1 + 0.1 * mmd + 0.1 * bce + 0.05 * kernel_prior + 0.05 * shared_bias
print("loss device", loss.device, float(loss))
loss.backward()
print("BACKWARD_OK")
for n, p in m.named_parameters():
    if p.grad is None:
        print("NO_GRAD", n)
print("done")
