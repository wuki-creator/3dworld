import sys, torch
sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
from model_world_h1_v14 import WorldModelH1V14
torch.manual_seed(421)
m = WorldModelH1V14(18533, d_model=256).to("cuda")
print("raw device", m.raw_kernel_alpha.device, "req", m.raw_kernel_alpha.requires_grad)
ka = m.kernel_alpha()
print("ka device", ka.device, ka.shape)
init = m.kernel_alpha_init
print("init device", init.device)
loss = ((ka - init) ** 2).sum()
print("loss device", loss.device)
loss.backward()
print("KERNEL_ONLY_OK grad", m.raw_kernel_alpha.grad)
import torch as t
m2 = WorldModelH1V14(18533, d_model=256).to("cuda")
ctrl = t.rand(4, 18533, device="cuda")
pert = t.randint(0, 18533, (4,), device="cuda")
pred = m2.predict_delta(ctrl, pert)
pred.sum().backward()
print("PRED_OK kernel grad", m2.raw_kernel_alpha.grad)
