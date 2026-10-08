import sys, torch
sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
from model_world_h1_v14 import WorldModelH1V14
torch.manual_seed(421)
G, DM, B = 18533, 256, 8
cfg = dict(n_genes=G, d_model=DM)
m = WorldModelH1V14(**cfg).to("cuda")
from train_magworld_h1_v14 import gaussian_mmd, sign_bce
ctrl = torch.rand(B, G, device="cuda") * 5.0
yb = torch.randn(B, G, device="cuda") * 0.01
pert = torch.randint(0, G, (B,), device="cuda")
mask = torch.rand(B, G, device="cuda") > 0.9
pred = m.predict_delta(ctrl, pert)
terms = {
 "l1": (pred - yb).abs().mean(1).mean(),
 "mmd": gaussian_mmd(pred, yb),
 "bce": sign_bce(pred, yb, mask, 0.05),
 "kernel": ((m.kernel_alpha().to("cuda") - m.kernel_alpha_init.to("cuda")) ** 2).sum(),
 "shared": m.effective_shared_bias_scale() * m.gene_bias.abs().mean(),
 "pred_only": pred.pow(2).mean(),
}
for name, t in terms.items():
    try:
        t.backward(retain_graph=True)
        print("OK", name, "device", t.device)
    except Exception as e:
        print("FAIL", name, str(e)[:120])
    m.zero_grad(set_to_none=True)
    pred = m.predict_delta(ctrl, pert)
    terms["mmd"] = gaussian_mmd(pred, yb)
