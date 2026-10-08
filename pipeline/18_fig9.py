# -*- coding: utf-8 -*-
"""Fig 9: graph-prior operators + world-model dynamics panels."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

OUT = "/nfs_beijing/zizhuo/vcc/results/magworld/figures_v2"
os.makedirs(OUT, exist_ok=True)
GP = "/nfs_beijing/zizhuo/vcc/results/magworld/graphprior"
WM = "/nfs_beijing/zizhuo/vcc/results/magworld/worldmodel"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

# ---------- a) GI-prior operators on Norman ----------
df = pd.read_csv(f"{GP}/graphprior_norman.csv")
order = ["MeanDelta", "ridge_on_E", "GI_PROP_SGC", "GI_PPR_diffusion",
         "GI_LAP_smoothedNW", "GCN_direct", "GCN_emb_ridge"]
df = df.set_index("model").loc[order]
# GEARS official comparison
gears_csv = f"{WM}/../gears/gears_metrics_test34.csv"
gears_row = None
if os.path.exists(gears_csv):
    gdf = pd.read_csv(gears_csv)
    gears_row = pd.DataFrame([{
        "model": "GEARS_official",
        "delta_pearson": gdf.delta_pearson.mean(),
        "delta_pearson_std": gdf.delta_pearson.std(),
        "specific_pearson": gdf.specific_pearson.mean(),
        "discrimination": gdf.disc_rank.mean(),
    }]).set_index("model")
    df = pd.concat([df, gears_row])
order = list(df.index)
x = np.arange(len(df))
ax = axes[0]
b1 = ax.bar(x - 0.2, df["delta_pearson"], width=0.38,
            color="#4878d0", label="delta Pearson r")
b2 = ax.bar(x + 0.2, df["specific_pearson"], width=0.38,
            color="#d65f5f", label="specific r")
ax.axhline(0.6141, color="#4878d0", ls="--", lw=1)
ax.axhline(0.0, color="#d65f5f", ls="--", lw=1)
ax.text(len(df) - 0.5, 0.618, "mean-delta\nbaseline", fontsize=7.5,
        color="#4878d0", ha="right")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", "\n") for m in df.index], fontsize=7.5)
ax.set_ylim(-0.12, 0.72)
ax.set_ylabel("held-out Pearson r (n=34 genes)")
ax.set_title("a. GI-graph priors do not unlock\nspecific predictability", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
cfg = json.load(open(f"{GP}/graphprior_config.json"))["graph"]
ax.text(0.02, 0.02,
        f"GI graph: {cfg['nodes']:,} nodes, {cfg['edges']:,} edges\n"
        "(regulons + co-expression + epistasis)",
        transform=ax.transAxes, fontsize=7, color="#555555")

# ---------- b) world-model velocity + future state ----------
vel = pd.read_csv(f"{WM}/velocity_prediction.csv").set_index("model")
fut = pd.read_csv(f"{WM}/future_state.csv").set_index("model")
ax = axes[1]
vorder = ["flow_mlp", "linear_ridge", "kNN_velocity", "mean_drift",
          "persistence_v0"]
vel = vel.loc[vorder]
x = np.arange(len(vel))
ax.bar(x, vel["cos"], color="#6acc64", width=0.55)
for i, v in enumerate(vel["cos"]):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(["flow MLP", "linear", "kNN", "mean\ndrift", "v=0"],
                   fontsize=8)
ax.set_ylabel("cosine(v_pred, v_true)  (held-out)")
ax.set_ylim(0, 1.0)
ax.set_title("b. Adapted-JEPA world-model base: latent\npredictive dynamics (pancreas, 32-d)",
             fontsize=10)

# inset: future-state MSE relative to persistence (each in its own latent
# space: flow field in VAE latent, JEPA predictor in JEPA latent)
axin = ax.inset_axes([0.13, 0.16, 0.82, 0.30])
ford = ["flow", "kNN", "linear", "mean_drift", "persistence"]
fv = fut.loc[ford, "future_mse"] / fut.loc["persistence", "future_mse"]
jcsv = f"{WM}/future_state_jepa.csv"
if os.path.exists(jcsv):
    jdf = pd.read_csv(jcsv).set_index("model")
    fv = pd.concat([fv, pd.Series(
        {"JEPA": jdf.loc["JEPA", "future_mse"]
         / jdf.loc["persistence", "future_mse"]})])
short = {"flow": "flow", "kNN": "kNN", "linear": "lin",
         "mean_drift": "mean", "persistence": "pers", "JEPA": "JEPA"}
colors = ["#e8a33d" if m == "JEPA" else "#b47cc7" for m in fv.index]
axin.bar(np.arange(len(fv)), fv, color=colors, width=0.55)
axin.axhline(1.0, color="#888888", ls="--", lw=0.8)
axin.set_xticks(np.arange(len(fv)))
axin.set_xticklabels([short[m] for m in fv.index], fontsize=6.5)
axin.set_ylim(0, float(fv.max()) * 1.3)
axin.set_title("future-state MSE (rel. to persistence)", fontsize=7)
axin.tick_params(labelsize=6.5)

# ---------- c) intervention rollout ----------
Z = np.load(f"{WM}/vae_latent.npy")
roll = np.load(f"{WM}/rollout_RELA.npy")     # (T+1, n0, 32)
pca = PCA(2).fit(Z)
Z2 = pca.transform(Z)
r2 = np.stack([pca.transform(roll[t]) for t in range(roll.shape[0])])
ax = axes[2]
ax.scatter(Z2[::6, 0], Z2[::6, 1], s=2, c="#cccccc", label="cells")
sub = np.random.default_rng(0).choice(roll.shape[1], 60, replace=False)
for j in sub:
    ax.plot(r2[:, j, 0], r2[:, j, 1], color="#4878d0", alpha=0.25, lw=0.8)
ax.scatter(r2[-1, :, 0].mean(), r2[-1, :, 1].mean(), s=60, c="#d65f5f",
           marker="*", label="action endpoint (mean)")
ax.set_title("c. In silico TF action rollout on\nthe learned dynamics (RELA)",
             fontsize=10)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=7.5, loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT}/Fig9_worldmodel.png", dpi=170, bbox_inches="tight")
print("saved", f"{OUT}/Fig9_worldmodel.png")
print("FIG9_DONE")
