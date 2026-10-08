# -*- coding: utf-8 -*-
"""Port of MagWorld (wuki-creator/Magtree) to the VCC 2026 H1 signature data.

Trains WorldModel (magnetic latent field + K-step rollout, xTrimo fused loss)
on h1_train_signatures (150 targets + 158 replicates) and evaluates on
h1_val_signatures (50 targets) with the same delta-space metrics used for v13:
cosine / fid / reach / jac / sign_acc on the true top-100 DE genes.

Gene embeddings are initialized with a PCA-64 projection of our hybrid 256-d
embeddings (MagWorld's generalization mechanism, bootstrapped).

Usage:
  python3 magworld_h1.py [--magnet 1] [--lambda_pds 0.05] [--epochs 200]
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

WORK = "/nfs_beijing_os/zizhuo_vcc/work"
sys.path.insert(0, WORK + "/magtree_src/src")
from model_world import WorldModel  # noqa: E402

TRAIN = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_train_signatures.npz"
VAL = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz"
EMB = "/nfs_beijing_os/zizhuo_vcc/embeddings/vcc_gene_embeddings_hybrid_256.npy"


def load_split(path, lookup):
    d = np.load(path, allow_pickle=False)
    genes = [str(g) for g in d["genes"]]
    items = []  # (target_idx_in_panel, ctrl, effect)
    for key in ("targets", "replicate_targets"):
        if key not in d.files:
            continue
        tgts = d[key].astype(str)
        ctrls = d["controls" if key == "targets" else "replicate_controls"]
        effs = d["effects" if key == "targets" else "replicate_effects"]
        for t, c, e in zip(tgts, ctrls, effs):
            if t in lookup:
                items.append((lookup[t], c.astype(np.float32), e.astype(np.float32)))
    return items


def pds_contrastive_loss(pred, y, x_ctrl, tau=0.2):
    dp = F.normalize(pred - x_ctrl, dim=-1)
    dt = F.normalize(y - x_ctrl, dim=-1)
    sim = dp @ dt.T / tau
    labels = torch.arange(sim.shape[0], device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


def evaluate_delta(delta_pred, eff, top_k=100):
    n = len(delta_pred)
    cos, fid_num, fid_den = [], [], []
    reach, jac, sign_accs = [], [], []
    for i in range(n):
        p, t = delta_pred[i], eff[i]
        cos.append(float((p * t).sum() / max(np.linalg.norm(p) * np.linalg.norm(t), 1e-8)))
        de = np.argpartition(np.abs(t), -top_k)[-top_k:]
        w = np.abs(t[de])
        agree = (np.sign(p[de]) == np.sign(t[de])) & (w > 0)
        fid_num.append((w * agree).sum())
        fid_den.append(w.sum())
        pred_top = np.argpartition(np.abs(p), -top_k)[-top_k:]
        inter = len(np.intersect1d(de, pred_top))
        reach.append(inter / top_k)
        jac.append(inter / (2 * top_k - inter))
        sign_accs.append(float((np.sign(p[de]) == np.sign(t[de])).mean()))
    return dict(cosine=float(np.mean(cos)),
                fid=float(np.mean(np.array(fid_num) / np.maximum(fid_den, 1e-8))),
                reach=float(np.mean(reach)), jac=float(np.mean(jac)),
                sign_acc=float(np.mean(sign_accs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnet", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_z", type=int, default=64)
    ap.add_argument("--d_hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--predict-delta", action="store_true",
                    help="train to predict the effect directly (v13-style) "
                         "instead of the post-perturbation mean")
    ap.add_argument("--de_weight", type=float, default=1.0)
    ap.add_argument("--lambda_pds", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--out", default="/nfs_beijing_os/zizhuo_vcc/ckpts/magworld_h1/magworld_h1.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtr = np.load(TRAIN, allow_pickle=False)
    genes = [str(g) for g in dtr["genes"]]
    lookup = {g: i for i, g in enumerate(genes)}
    G = len(genes)

    train_items = load_split(TRAIN, lookup)
    val_items = load_split(VAL, lookup)
    print("train conds=%d val conds=%d G=%d" % (len(train_items), len(val_items), G), flush=True)

    # PCA-64 init of gene embeddings from hybrid 256-d embeddings
    emb = np.load(EMB).astype(np.float32)
    emb = emb - emb.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(emb, full_matrices=False)
    pca64 = (u[:, :64] * s[:64]).astype(np.float32)
    pca64 = pca64 / (np.abs(pca64).std() + 1e-8) * 0.05

    model = WorldModel(G, d_model=64, d_z=args.d_z, d_hidden=args.d_hidden,
                       n_steps=args.steps,
                       use_magnet=bool(args.magnet)).to(device)
    with torch.no_grad():
        model.gene_emb.weight.copy_(torch.from_numpy(pca64).to(device))
    print("params: %.0fk" % (sum(p.numel() for p in model.parameters()) / 1e3), flush=True)

    X = torch.from_numpy(np.stack([c for _, c, _ in train_items])).to(device)
    if args.predict_delta:
        Y = torch.from_numpy(np.stack([e for _, _, e in train_items])).to(device)
    else:
        Y = torch.from_numpy(np.stack([c + e for _, c, _ in train_items])).to(device)
    pidx_all = [t for t, _, _ in train_items]
    ctrl_mean = X.mean(0)

    # DE weights: top-50 by |effect| per condition (xTrimo style)
    de_masks = []
    for _, c, e in train_items:
        m = torch.ones(G, device=device)
        top = torch.argsort(-torch.from_numpy(np.abs(e)).to(device))[:50]
        m[top] = 1.0 + args.de_weight
        de_masks.append(m)
    W = torch.stack(de_masks)

    Xv = torch.from_numpy(np.stack([c for _, c, _ in val_items])).to(device)
    Yv = torch.from_numpy(np.stack([c + e for _, c, _ in val_items])).to(device)
    Ev = np.stack([e for _, _, e in val_items])
    pidx_v = [torch.tensor([t]) for t in [t for t, _, _ in val_items]]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n = X.shape[0]
    steps_per_ep = int(np.ceil(n / args.batch))
    t0 = time.time()
    best = (-1e9, None, -1)
    bad = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for s in range(steps_per_ep):
            b = perm[s * args.batch:(s + 1) * args.batch]
            if len(b) < 4:
                continue
            xb, yb, wb = X[b], Y[b], W[b]
            # MagWorld convention: list of per-sample index tensors
            pidx = [torch.tensor([pidx_all[i]]) for i in b]
            pred = model(xb, pidx)
            if args.predict_delta:
                t_delta = yb  # Y already holds the effect
                l_mae = (wb * (pred - t_delta).abs()).mean()
                dp = F.normalize(pred, dim=-1)
                dt = F.normalize(t_delta, dim=-1)
                sim = dp @ dt.T / args.tau
                labels = torch.arange(sim.shape[0], device=sim.device)
                l_pds = 0.5 * (F.cross_entropy(sim, labels)
                               + F.cross_entropy(sim.T, labels))
            else:
                l_mae = (wb * (pred - yb).abs()).mean()
                l_pds = pds_contrastive_loss(pred, yb, xb, args.tau)
            loss = l_mae + args.lambda_pds * l_pds
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        # val
        model.eval()
        with torch.no_grad():
            pred_v = model(Xv, pidx_v)
            delta_v = pred_v.cpu().numpy() if args.predict_delta else (pred_v - Xv).cpu().numpy()
        m = evaluate_delta(delta_v, Ev)
        score = m["fid"] + m["cosine"]
        flag = ""
        if score > best[0]:
            best = (score, {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}, ep)
            bad = 0
            flag = " *best*"
        else:
            bad += 1
        if ep % 10 == 0 or flag:
            print("epoch %3d loss=%.4f | val cos=%.4f fid=%.4f reach=%.4f "
                  "jac=%.4f sign_acc=%.4f%s (%.0fs)"
                  % (ep, tot / steps_per_ep, m["cosine"], m["fid"], m["reach"],
                     m["jac"], m["sign_acc"], flag, time.time() - t0), flush=True)
        if bad >= args.patience:
            print("early stop at %d" % ep, flush=True)
            break

    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pred_v = model(Xv, pidx_v)
        delta_v = pred_v.cpu().numpy() if args.predict_delta else (pred_v - Xv).cpu().numpy()
    m = evaluate_delta(delta_v, Ev)
    print("FINAL best_epoch=%d val: %s" % (best[2], json.dumps(m)), flush=True)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "genes": genes,
                "config": {"n_genes": G, "d_model": 64, "d_z": args.d_z,
                           "d_hidden": args.d_hidden,
                           "n_steps": args.steps, "use_magnet": bool(args.magnet)},
                "best_epoch": best[2], "val": m, "args": vars(args)}, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
