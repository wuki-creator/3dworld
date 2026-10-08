# -*- coding: utf-8 -*-
"""Local small-data test for v34/v35/v36/v37 against the v25 champion base.

Tests:
  1. forward shape + no NaN
  2. init equivalence: v34/v35/v36/v37 == v25 at init (zero-init discipline)
  3. gradient flow to new params after one backward
  4. v35 pole sign rule: flipping poles flips the deviation from v25
  5. v37 kq force: flipping kq_k output negates the deviation from v25
  6. v36: after unzeroing head, chain is active and causal-mask rows are nan-free
"""
import numpy as np
import torch.nn.functional as F
import torch

import model_world_h1_v25 as m25
import model_world_h1_v34 as m34
import model_world_h1_v35 as m35
import model_world_h1_v36 as m36
import model_world_h1_v37 as m37

G = 48
B = 8
KW = dict(n_genes=G, d_model=64, d_z=32, d_hidden=48, d_dir=16,
          graph_k=8, n_layers=3, response_rank=16)

# tiny prior graph (DGIDB-like npz)
rng = np.random.RandomState(0)
ei = rng.randint(0, G, size=(2, 120))
es = rng.choice([-1.0, 1.0], size=120).astype(np.float32)
ew = rng.rand(120).astype(np.float32)
np.savez("tiny_prior.npz", edge_index=ei, edge_sign=es, edge_w=ew)

torch.manual_seed(7)
x_ctrl = torch.rand(B, G) * 3.0
pert = torch.tensor([3, 7, 11, 15, 19, 23, 31, 40])


def build(cls):
    torch.manual_seed(42)  # identical base init across models
    return cls(prior_graph_paths=["tiny_prior.npz"], **KW)


models = {
    "v25": build(m25.WorldModelH1V25),
    "v34": build(m34.WorldModelH1V34),
    "v35": build(m35.WorldModelH1V35),
    "v36": build(m36.WorldModelH1V36),
    "v37": build(m37.WorldModelH1V37),
}

print("== 1. forward shape / NaN check")
outs = {}
for name, model in models.items():
    with torch.no_grad():
        out = model.predict_delta(x_ctrl, pert)
    outs[name] = out
    assert out.shape == (B, G), (name, out.shape)
    assert torch.isfinite(out).all(), name
    print("  %-4s out[0,:4] = %s  |max|=%.4f" %
          (name, np.round(out[0, :4].numpy(), 4), float(out.abs().max())))

print("== 2. init equivalence vs v25 (zero-init discipline)")
for name in ("v34", "v35", "v36", "v37"):
    diff = (outs[name] - outs["v25"]).abs().max().item()
    status = "OK " if diff < 1e-5 else "FAIL"
    print("  %s %-4s max|diff| = %.2e" % (status, name, diff))
    assert diff < 1e-5, name

print("== 3. gradient flow")
for name in ("v34", "v35", "v36", "v37"):
    model = models[name]
    out = model.predict_delta(x_ctrl, pert)
    loss = (out ** 2).mean()
    loss.backward()
    if name == "v34":
        # GAT 权重经零初始化注入头，初始梯度被挡（设计使然）；
        # 先动的是三条注入头，它们通了之后 GAT 才学
        p = model.prior_charge.weight
        p2 = model.gat[0].q.weight
        print("  v34 gat.q |grad| = %.2e (0 expected: 零初始化注入头挡着)"
              % float(p2.grad.abs().sum()))
    elif name == "v35":
        p = model.raw_resp  # gate must move first
        p2 = model.raw_pole  # blocked at init (expected 0)
        print("  v35 raw_pole.grad abs sum = %.2e (0 expected: gate未开)"
              % float(p2.grad.abs().sum()))
    elif name == "v36":
        p = model.tf_head.weight
    else:
        p = model.gene_axis
    gn = float(p.grad.abs().sum())
    ok = gn > 0 and torch.isfinite(p.grad).all().item()
    print("  %s %-4s new-param |grad| = %.4e" % ("OK " if ok else "FAIL", name, gn))
    assert ok, name
    model.zero_grad()

print("== 4. v35 pole sign rule（全局对称性：全极性翻转 → 响应不变）")
mdl = models["v35"]
with torch.no_grad():
    mdl.raw_resp.fill_(2.0)  # open the gate
base = mdl.predict_delta(x_ctrl, pert)
with torch.no_grad():
    # 全部基因极性取反：S ∝ pole 反号、resp = tanh(pole·S_n/τ) 不变
    # （同极/异极是相对关系 —— 这正是「同极↓异极↑」的数学表达）
    mdl.raw_pole.neg_()
flipped = mdl.predict_delta(x_ctrl, pert)
sym = float((base - flipped).abs().max())
print("  global-flip |out1-out2| = %.2e (≈0 期望)" % sym)
assert sym < 1e-4, "pole rule broken: response changed under global flip"
# 再验证非对称翻转确实改变响应（极性有实际作用）
with torch.no_grad():
    mdl.raw_pole.neg_()                      # 还原
    mdl.raw_pole[pert] = -mdl.raw_pole[pert]  # 只翻源
asym = mdl.predict_delta(x_ctrl, pert)
dev = float((asym - base).abs().max())
print("  source-only flip |diff| = %.2e (>0 期望: 源极性有作用)" % dev)
assert dev > 1e-6

print("== 5. v37 vector-force sign（kq 链闭式检验）")
mdl = models["v37"]
with torch.no_grad():
    mdl.gene_axis.copy_(torch.randn_like(mdl.gene_axis) * 0.5)  # 打开通道


def kq_dx_tf(model):
    """按 v37 公式手工复算 dx_tf（单次 kq 力链，独立于磁流循环）。"""
    with torch.no_grad():
        e = model.gene_emb.weight
        ctx = F.layer_norm(x_ctrl, (G,)) if model.normalize_context else x_ctrl
        z = model.context_encoder(ctx)
        s = model.state_gene(e)[None, :, :] + model.state_ctx(z)[:, None, :]
        src_vec = torch.tanh(model.src_tf(torch.cat([e[pert], z], dim=-1)))
        q = model.kq_q(s)
        k = model.kq_k(src_vec)
        E = k[:, None, :] * q
        v = model.kq_v(s)
        u = E * v
        axis = model.gene_axis / (1.0 + model.gene_axis.norm(dim=-1, keepdim=True))
        return (u * axis[None]).sum(-1)


dx_base = kq_dx_tf(mdl)
with torch.no_grad():
    mdl.kq_k.weight.neg_()   # k -> -k: E 精确反号
dx_flip = kq_dx_tf(mdl)
res = float((dx_base + dx_flip).abs().max())
scale = float(dx_base.abs().max()) + 1e-12
print("  kq antisymmetry residual = %.2e (rel %.2e)" % (res, res / scale))
assert res / scale < 1e-4, "kq force sign broken"
with torch.no_grad():
    mdl.kq_k.weight.neg_()  # 还原
# 端到端：通道确实影响输出
with torch.no_grad():
    saved = mdl.gene_axis.clone()
    mdl.gene_axis.zero_()
out_closed = mdl.predict_delta(x_ctrl, pert)
with torch.no_grad():
    mdl.gene_axis.copy_(saved)
out_open = mdl.predict_delta(x_ctrl, pert)
eff = float((out_open - out_closed).abs().max())
print("  channel effect |out_open-out_closed| = %.2e (>0 期望)" % eff)
assert eff > 1e-6
print("  OK  向量力方向随 k 精确翻转，端到端通道生效")

print("== 6. v36 chain activity after unzeroing head")
mdl = models["v36"]
with torch.no_grad():
    mdl.tf_head.weight.normal_(0, 0.1)
out = mdl.predict_delta(x_ctrl, pert)
assert torch.isfinite(out).all()
diff = (out - outs["v25"]).abs().max().item()
print("  |out - v25| max = %.4e (>0 => 因果链生效)" % diff)
assert diff > 1e-6
diag = mdl.magnetic_diagnostics(pert)
print("  diag:", {k: v for k, v in diag.items()
                  if k in ("pole_pos_frac", "beta_attract", "tf_head_w")})

print("\nALL LOCAL TESTS PASSED")
