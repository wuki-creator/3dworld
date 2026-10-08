# -*- coding: utf-8 -*-
"""Perturbation-factor normalization network (post-hoc delta calibrator).

Learns a per-gene gate s(pert, gene) in [-2, 2] applied to the raw predicted
delta, plus a per-perturbation L1 normalization of the delta vector:

    z = delta / (mean|delta| + eps) * norm_scale          (perturbation factor normalization)
    s = 2 * tanh(MLP([z, |delta|, gene_emb, target_emb])) (per-gene gate, CAN FLIP SIGNS)
    calibrated = s * delta

Trained on the H1 train signatures (150 perts), early-stopped on the val
split (50 perts). Loss = cosine + weighted sign agreement on top-100 true DE
genes + weighted L1 -- i.e. exactly the quantity the platform fid rewards.

Usage:
  python3 pert_norm_net.py --ckpt <member.pt> [--epochs 300] [--out ckpt]
"""
import argparse
import importlib
import json
import sys

import numpy as np
import torch
import torch.nn as nn

WORK = "/nfs_beijing_os/zizhuo_vcc/work"
sys.path.insert(0, WORK)
from proxy_score import load_member, raw_delta, delta_metrics  # noqa: E402

TRAIN = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_train_signatures.npz"
VAL = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz"
EMB = "/nfs_beijing_os/zizhuo_vcc/embeddings/vcc_gene_embeddings_hybrid_256.npy"


class PertNormNet(nn.Module):
    """Per-gene gated normalization of perturbation factors."""

    def __init__(self, d_emb=256, d_proj=8, d_hidden=64, n_genes=18533):
        super().__init__()
        self.emb_proj = nn.Linear(d_emb, d_proj)
        self.mlp = nn.Sequential(
            nn.Linear(2 + 2 * d_proj, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        # per-gene bias scale so frequent flips don't fight the MLP
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        self.norm_scale = nn.Parameter(torch.ones(1))

    def forward(self, delta, gene_emb, target_emb):
        # delta: (B, G) raw predicted LFC; gene_emb: (G, d_emb); target_emb: (B, d_emb)
        B, G = delta.shape
        l1 = delta.abs().mean(1, keepdim=True)
        z = delta / (l1 + 1e-6) * self.norm_scale                      # (B, G)
        eg = self.emb_proj(gene_emb)                                    # (G, P)
        et = self.emb_proj(target_emb).unsqueeze(1).expand(B, G, -1)    # (B, G, P)
        feats = torch.cat([
            z.unsqueeze(-1),
            delta.abs().unsqueeze(-1),
            eg.unsqueeze(0).expand(B, G, -1),
            et,
        ], dim=-1)                                                      # (B, G, 2+2P)
        s = 2.0 * torch.tanh(self.mlp(feats).squeeze(-1) + self.gene_bias)
        return s * delta


def topk_sign_loss(cal, eff, top_k=100):
    """Weighted sign agreement on the true top-k DE genes (platform fid proxy)."""
    B, G = eff.shape
    losses = []
    for i in range(B):
        idx = torch.topk(eff[i].abs(), top_k).indices
        w = eff[i, idx].abs()
        w = w / w.sum().clamp_min(1e-8)
        agree = torch.sigmoid(cal[i, idx] * eff[i, idx] * 20.0)  # sharp sign match
        losses.append((w * agree).sum())
    return -torch.stack(losses).mean()


def weighted_l1(cal, eff, top_k=200):
    B = eff.shape[0]
    losses = []
    for i in range(B):
        idx = torch.topk(eff[i].abs(), top_k).indices
        w = eff[i, idx].abs()
        w = w / w.sum().clamp_min(1e-8)
        losses.append((w * (cal[i, idx] - eff[i, idx]).abs()).sum())
    return torch.stack(losses).mean()


def cosine_loss(cal, eff):
    num = (cal * eff).sum(1)
    den = cal.norm(dim=1) * eff.norm(dim=1)
    return (1.0 - num / den.clamp_min(1e-8)).mean()


def evaluate(cal_np, eff_np, top_k=100):
    m = delta_metrics(cal_np, eff_np, None, top_k)
    cos = float(((cal_np * eff_np).sum(1) / np.maximum(
        np.linalg.norm(cal_np, axis=1) * np.linalg.norm(eff_np, axis=1), 1e-8)).mean())
    # sign accuracy on true top-100
    accs = []
    for i in range(len(cal_np)):
        idx = np.argpartition(np.abs(eff_np[i]), -100)[-100:]
        accs.append(float((np.sign(cal_np[i, idx]) == np.sign(eff_np[i, idx])).mean()))
    m["cosine"] = cos
    m["sign_acc_top100"] = float(np.mean(accs))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--w-sign", type=float, default=2.0)
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-cos", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="/nfs_beijing_os/zizhuo_vcc/ckpts/pertnorm_v13s113.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genes_emb = torch.from_numpy(np.load(EMB).astype(np.float32)).to(device)
    n_genes = genes_emb.shape[0]

    model, genes = load_member(args.ckpt, device)
    lookup = {g: i for i, g in enumerate(genes)}

    def load_split(path):
        v = np.load(path, allow_pickle=False)
        targets = v["targets"].astype(str)
        ok = np.array([t in lookup for t in targets])
        tidx = np.asarray([lookup[t] for t in targets[ok]], dtype=np.int64)
        eff = v["effects"].astype(np.float32)[ok]
        ctrl = v["controls"].astype(np.float32)[ok].mean(0)
        raw = raw_delta(model, ctrl, tidx, device)
        return (torch.from_numpy(raw).to(device),
                torch.from_numpy(eff).to(device),
                torch.from_numpy(genes_emb[tidx].cpu().numpy()).to(device))

    raw_tr, eff_tr, temb_tr = load_split(TRAIN)
    raw_va, eff_va, temb_va = load_split(VAL)

    net = PertNormNet(d_emb=genes_emb.shape[1], n_genes=n_genes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = (-1e9, None, -1)
    for epoch in range(1, args.epochs + 1):
        net.train()
        opt.zero_grad()
        cal = net(raw_tr, genes_emb, temb_tr)
        loss = (args.w_cos * cosine_loss(cal, eff_tr)
                + args.w_sign * topk_sign_loss(cal, eff_tr)
                + args.w_l1 * weighted_l1(cal, eff_tr))
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            net.eval()
            with torch.no_grad():
                cal_va = net(raw_va, genes_emb, temb_va).cpu().numpy()
            m = evaluate(cal_va, eff_va.cpu().numpy())
            score = m["fid"] + m["cosine"]  # composite: fidelity + overall shape
            flag = ""
            if score > best[0]:
                best = (score, {kk: vv.detach().cpu().clone()
                                for kk, vv in net.state_dict().items()}, epoch)
                flag = " *best*"
            print("epoch %4d loss=%.4f | val cos=%.4f fid=%.4f reach=%.4f "
                  "jac=%.4f sign_acc=%.4f nmae=%.4f%s"
                  % (epoch, loss.item(), m["cosine"], m["fid"], m["reach"],
                     m["jac"], m["sign_acc_top100"], m["nmae"], flag), flush=True)

    # baseline comparison (raw delta, no calibration)
    m_raw = evaluate(raw_va.cpu().numpy(), eff_va.cpu().numpy())
    print("BASELINE raw val: cos=%.4f fid=%.4f reach=%.4f jac=%.4f sign_acc=%.4f"
          % (m_raw["cosine"], m_raw["fid"], m_raw["reach"], m_raw["jac"],
             m_raw["sign_acc_top100"]), flush=True)

    if best[1] is not None:
        torch.save({"state_dict": best[1], "genes": genes,
                    "best_epoch": best[2], "val_score": best[0],
                    "args": vars(args)}, args.out)
        print("saved %s (best epoch %d, val score %.4f)" % (args.out, best[2], best[0]))
    with open(args.out + ".report.json", "w") as f:
        json.dump({"baseline_val": m_raw, "best_epoch": best[2],
                   "best_val_score": best[0]}, f, indent=2)


if __name__ == "__main__":
    main()
