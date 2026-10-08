import sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
from model_world_h1_v14 import WorldModelH1V14
torch.manual_seed(0)
G, DM, B = 18533, 256, 4
m = WorldModelH1V14(G, d_model=DM).cuda()
ctrl = torch.rand(B, G, device="cuda") * 5.0
pert = torch.randint(0, G, (B,), device="cuda")
with torch.no_grad():
    e = m.gene_emb.weight
    pos = F.normalize(m.pole(e), dim=-1)
    charge = F.softplus(m.charge(e).squeeze(-1) + m.charge_bias)
    nb = m._knn_edges(pos)
    p_nb = pos[nb]
    diff = pos[:, None, :] - p_nb
    r_nb = diff.norm(dim=-1).clamp_min(1e-4)
    k_raw = m._kernel(r_nb)
    k_nb = k_raw / (k_raw.mean(-1, keepdim=True) + 1e-6)
    print("r_nb min/mean/max", float(r_nb.min()), float(r_nb.mean()), float(r_nb.max()))
    print("k_raw min/mean/max", float(k_raw.min()), float(k_raw.mean()), float(k_raw.max()))
    print("k_nb min/mean/max", float(k_nb.min()), float(k_nb.mean()), float(k_nb.max()))
    indices = pert.reshape(-1).long()
    kappa = F.softplus(m.raw_source_kappa)
    p_src = pos[indices]
    src_diff = pos[None, :, :] - p_src[:, None, :]
    d_src = src_diff.norm(dim=-1).clamp_min(1e-4)
    ks_raw = m._kernel(d_src)
    k_src = ks_raw / (ks_raw.mean(dim=1, keepdim=True) + 1e-6)
    print("d_src min", float(d_src.min()))
    print("ks_raw max", float(ks_raw.max()), "k_src max", float(k_src.max()), "k_src mean", float(k_src.mean()))
    phi = kappa * charge[indices][:, None] * k_src
    print("phi min/mean/max", float(phi.min()), float(phi.mean()), float(phi.max()))
    r = (m.response_right(phi) @ m.response_left.weight) / (64 ** 0.5)
    print("r min/mean/max", float(r.min()), float(r.mean()), float(r.max()))
    print("tanh(r) abs max", float(torch.tanh(r).abs().max()))
    ctx_in = F.layer_norm(ctrl, (G,))
    z = m.context_encoder(ctx_in)
    s = m.state_gene(e)[None, :, :] + m.state_ctx(z)[:, None, :]
    a = torch.sigmoid(m.amp_gene(s).squeeze(-1) + m.amp_phi(phi.unsqueeze(-1)).squeeze(-1) + m.amp_v(phi.new_zeros(B, G).unsqueeze(-1)).squeeze(-1) + m.amp_bias)
    print("a min/mean/max", float(a.min()), float(a.mean()), float(a.max()))
    d0 = F.normalize(m.dir_proj(s), dim=-1)
    print("d0 shape", tuple(d0.shape))
    dx = a * torch.tanh(r) * torch.sigmoid(m.gate_bias)
    print("dx abs max", float(dx.abs().max()), "h after 3 layers approx", float((3 * dx).abs().max()))
    d = m.predict_delta(ctrl, pert)
    print("d min/mean/max", float(d.min()), float(d.mean()), float(d.max()))
    print("d square mean", float(d.pow(2).mean()))
