# -*- coding: utf-8 -*-
"""Curie-Weiss scaling experiment for the v14 magnetic-flow model.

Fix the perturbation set, inject Gaussian noise of strength sigma into the
control state, and measure the effective susceptibility

    chi(sigma) = mean ||D(x + sigma*eps) - D(x)||_2 / sigma

A physics-style linear-response model predicts chi ~ const (Curie) or a
divergence chi^-1 -> 0 at some critical sigma (Curie-Weiss). A generic MLP
has no such scaling law: its effective slope shrinks with noise, so chi^-1
grows with sigma.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/nfs_beijing_os/zizhuo_vcc/work")
from model_world_h1_v14 import WorldModelH1V14

CKPT = "/nfs_beijing_os/zizhuo_vcc/ckpts/v14e_seed421/best.pt"
VAL = "/nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz"
OUT = "/nfs_beijing_os/zizhuo_vcc/work/curie_weiss_v14"
DEVICE = "cuda"
SEED = 7
REPS = 8


def main():
    torch.manual_seed(SEED)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = WorldModelH1V14(**dict(ck["model_config"]))
    model.load_state_dict(ck["model_state"])
    model = model.to(DEVICE).eval()

    v = np.load(VAL, allow_pickle=False)
    genes = [str(g) for g in ck["genes"]]
    lookup = {g: i for i, g in enumerate(genes)}
    targets = v["targets"].astype(str)
    ok = np.array([t in lookup for t in targets])
    tidx = np.asarray([lookup[t] for t in targets[ok]], dtype=np.int64)
    ctrl = torch.from_numpy(v["controls"].astype(np.float32)[ok].mean(0)).to(DEVICE)
    base = torch.from_numpy(tidx).to(DEVICE)

    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    with torch.no_grad():
        clean = model.predict_delta(ctrl[None, :].expand(len(tidx), -1), base)
        chi, chi_std = [], []
        for s in sigmas:
            if s == 0.0:
                chi.append(np.nan)
                chi_std.append(np.nan)
                continue
            reps = []
            for _ in range(REPS):
                eps = torch.randn_like(ctrl)
                noisy = (ctrl + s * eps).clamp_min(0.0)
                d = model.predict_delta(noisy[None, :].expand(len(tidx), -1), base)
                resp = ((d - clean).norm(dim=1) / s).mean().item()
                reps.append(resp)
            chi.append(float(np.mean(reps)))
            chi_std.append(float(np.std(reps)))

    inv = [1.0 / c if c == c and c > 0 else np.nan for c in chi]
    np.savez(OUT + ".npz", sigmas=np.array(sigmas), chi=np.array(chi),
             chi_std=np.array(chi_std), chi_inv=np.array(inv))
    with open(OUT + ".json", "w") as fh:
        json.dump({"sigmas": sigmas, "chi": chi, "chi_inv": inv}, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].errorbar(sigmas[1:], chi[1:], yerr=chi_std[1:], marker="o")
    axes[0].set_xlabel("noise strength sigma")
    axes[0].set_ylabel("effective susceptibility chi")
    axes[0].set_title("Curie-Weiss test: chi(sigma)")
    axes[1].plot(sigmas[1:], inv[1:], marker="s", color="crimson")
    fit = np.polyfit(sigmas[1:], inv[1:], 1)
    xs = np.linspace(0, max(sigmas[1:]), 50)
    axes[1].plot(xs, np.polyval(fit, xs), "--", color="gray",
                 label="linear fit slope=%.3f" % fit[0])
    axes[1].set_xlabel("noise strength sigma")
    axes[1].set_ylabel("chi^-1")
    axes[1].set_title("inverse susceptibility (divergence = physics)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=140)
    print("CURIE_WEISS_OK", json.dumps({"chi": chi, "chi_inv": inv}))


if __name__ == "__main__":
    main()
