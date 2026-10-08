# -*- coding: utf-8 -*-
"""Fine-tune a sci-Plex-pretrained MagneticCrossCellV2 on H1 signatures.

Compatibility bridge: the pretrained model conditions the perturbation through
a *drug* embedding table.  H1 is CRISPRi (gene-target), so the drug table is
rebuilt with one slot per target gene and initialised from that gene's token
embedding, keeping the field geometry consistent across the two domains.  All
other weights transfer from the sci-Plex pretrain checkpoint.  Early stopping
with patience on a held-out target split; the best checkpoint is kept.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/zizhuo/cross_cell_vcc_v2/src")
from magnetic_cross_cell_v2 import MagneticCrossCellV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", required=True)
    p.add_argument("--signatures", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--loops", type=int, default=20)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def masked_mse(pred, truth, mask):
    err = (pred - truth) * mask
    return float((err ** 2).sum() / max(mask.sum(), 1e-8))


def cosine_row(pred, truth):
    num = (pred * truth).sum(1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    return float((num / np.maximum(den, 1e-8)).mean())


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    d = np.load(a.signatures, allow_pickle=False)
    genes = d["genes"].astype(str)
    targets = d["targets"].astype(str)
    x = d["controls"].astype(np.float32)
    y = d["effects"].astype(np.float32)
    mask = ((np.abs(y) > 0) | (x > 0)).astype(np.float32)

    ck = torch.load(a.pretrain, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    if cfg["n_genes"] != len(genes):
        raise ValueError("gene panel mismatch between pretrain and H1")
    if list(np.asarray(ck["genes"]).astype(str)) != list(genes):
        raise ValueError("gene order mismatch between pretrain and H1")

    n_t = len(targets)
    gene_rows = np.array([int(np.flatnonzero(genes == t)[0]) for t in targets], dtype=np.int64)

    model = MagneticCrossCellV2(
        cfg["n_genes"], n_t, latent=cfg["latent"], hidden=cfg["hidden"],
        heads=cfg["heads"], mlp_layers=cfg["mlp_layers"],
        field_layers=cfg["field_layers"], rank=cfg["rank"],
        max_loops=cfg["max_loops"]).to(device)

    # transfer compatible weights; rebuild drug table from gene embeddings
    model_sd = model.state_dict()
    transferred, skipped = [], []
    for key, value in ck["model_state"].items():
        if key in model_sd and tuple(model_sd[key].shape) == tuple(value.shape):
            model_sd[key] = value
            transferred.append(key)
        else:
            skipped.append(key)
    new_drugs = torch.zeros((n_t + 1, cfg["latent"]))
    src_genes = ck["model_state"]["genes.weight"]
    for i, g in enumerate(gene_rows):
        new_drugs[i + 1] = src_genes[g + 1]
    model_sd["drugs.weight"] = new_drugs
    model.load_state_dict(model_sd)
    print("transferred %d tensors, skipped %s" % (len(transferred), skipped), flush=True)

    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    mt = torch.from_numpy(mask).to(device)
    gt = torch.from_numpy(gene_rows + 1).to(device)          # gene slot: own target
    dt = torch.arange(1, n_t + 1, device=device)             # drug slot: own slot
    dose = torch.ones(n_t, 1, device=device)

    perm = rng.permutation(n_t)
    n_val = max(5, int(round(a.val_fraction * n_t)))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    print("train targets %d, val targets %d" % (len(train_idx), len(val_idx)), flush=True)

    def evaluate(indices):
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(indices), a.batch):
                b = torch.tensor(indices[start:start + a.batch], device=device)
                pred, _ = model.rollout(xt[b], gt[b], dt[b], dose.expand(len(b), 1), a.loops)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    best = {"cosine": -1.0}
    best_epoch = 0
    stale = 0
    history = []
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        totals = []
        for b in np.array_split(train_idx[rng.permutation(len(train_idx))],
                                max(1, int(np.ceil(len(train_idx) / a.batch)))):
            bt = torch.tensor(b, device=device)
            depth = int(rng.choice([10, 20, 30]))
            loss, mse, regsi = model.loss(xt[bt], yt[bt], gt[bt], dt[bt],
                                          dose.expand(len(bt), 1), mt[bt], depth)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss at epoch %d" % epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals.append([loss.item(), mse.item(), regsi.item()])
        vp = evaluate(val_idx)
        vm = masked_mse(vp, y[val_idx], mask[val_idx])
        vc = cosine_row(vp, y[val_idx])
        row = {"epoch": epoch, "train_loss": float(np.mean([t[0] for t in totals])),
               "val_mse": vm, "val_cosine": vc, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row), flush=True)
        if vc > best["cosine"]:
            best = {"cosine": vc, "mse": vm}
            best_epoch = epoch
            stale = 0
            torch.save({"model_state": model.state_dict(), "model_config": model.config,
                        "genes": genes.tolist(), "targets": targets.tolist(),
                        "gene_rows": gene_rows.tolist(), "loops": a.loops,
                        "best_epoch": best_epoch, "val": best,
                        "pretrain": str(a.pretrain)}, str(a.out))
        else:
            stale += 1
            if stale >= a.patience:
                print("early stop at epoch %d (best %d)" % (epoch, best_epoch), flush=True)
                break

    out = {"best_epoch": best_epoch, "best": best, "train_targets": len(train_idx),
           "val_targets": len(val_idx), "history_tail": history[-5:],
           "transferred": len(transferred), "skipped": skipped}
    with open(str(a.out) + ".report.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
