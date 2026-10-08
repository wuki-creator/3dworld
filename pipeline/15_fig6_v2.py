# -*- coding: utf-8 -*-
"""Fig6 v2: 多尺度基准总览 (2x3 面板)。
a 特异分量相关 (3 尺度) / b 判别排名 / c 单细胞留出 /
d 去噪 MSE 增益 / e 组合预测 / f 协同分布
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 9
BASE = "/nfs_beijing/zizhuo/vcc/results/magworld"
OUT = f"{BASE}/figures_v2"
import os
os.makedirs(OUT, exist_ok=True)

sing = pd.read_csv(f"{BASE}/norman/norman_singles_summary.csv")
tra = pd.read_csv(f"{BASE}/transfer/transfer_singles_summary.csv")
tdisc = pd.read_csv(f"{BASE}/transfer/transfer_discrimination_summary.csv")
pc = pd.read_csv(f"{BASE}/percell/percell_heldout_summary.csv")
den = pd.read_csv(f"{BASE}/denoise/denoise_by_size.csv")
ndb = pd.read_csv(f"{BASE}/norman/norman_doubles_summary.csv")
syn = pd.read_csv(f"{BASE}/norman/norman_synergy.csv")

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))

# a) 特异分量相关: 三尺度 (dat17 ridge / norman hybrid / transfer stack)
ax = axes[0, 0]
vals = [0.046, tra.set_index("model").loc["MAGWorld-stack",
        "specific_pearson"], np.nan]
# dat17 用 v5 实测 specific 近似 0 (置换验证); 画 norman 各模型 + transfer
models = ["MAGWorld-ridge", "MAGWorld-hybrid", "MAGWorld-factor",
          "MeanDelta"]
v1 = sing.set_index("model").reindex(models)["specific_pearson"] \
    .fillna(0).values
v2 = tra.set_index("model").reindex(
    ["MAGWorld-factor-T2k30", "MAGWorld-stack", "MeanDelta-norman68"]
)["specific_pearson"].fillna(0).values
x = np.arange(4)
w = 0.38
ax.bar(x - w/2, [v1[0], v1[1], v1[2], 0], w, label="Norman (68 train)",
       color="#4878d0")
ax.bar(x + w/2, [v2[0], v2[1], v2[2], 0], w,
       label="Replogle transfer (2,057 train)", color="#d65f5f")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["ridge", "hybrid", "factor", "mean"], rotation=20)
ax.set_ylabel("specific-component Pearson r")
ax.set_title("a. Gene-specific structure $\\approx$ 0 at all scales")
ax.legend(fontsize=7.5, loc="upper right")

# b) 判别排名
ax = axes[0, 1]
d = tdisc.set_index("model")
names = ["MAGWorld-NW-T3t0.4", "MAGWorld-factor-T2k30", "MAGWorld-stack",
         "MeanDelta-norman68"]
ax.bar(range(len(names)), d.reindex(names)["rank"], color="#6acc64")
ax.axhline(17.5, color="r", ls="--", lw=1,
           label="random chance (n=34)")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(["NW", "factor", "stack", "mean"], rotation=20)
ax.set_ylabel("mean discrimination rank (lower = better)")
ax.set_title("b. Perturbation discrimination at chance")
ax.legend(fontsize=7.5)

# c) 单细胞留出
ax = axes[0, 2]
names = ["MeanDelta-shift", "GlobalMean-shift", "Identity",
         "MAGWorld-cVAE"]
v = pc.set_index("model").reindex(names)
x = np.arange(len(names))
ax.bar(x, v["percell_pearson"], 0.55, color="#4878d0",
       label="per-cell Pearson")
ax2 = ax.twinx()
ax2.plot(x, v["delta_pearson"], "o-", color="#d65f5f",
         label="delta Pearson")
ax2.set_ylabel("held-out delta Pearson r", color="#d65f5f")
ax.set_xticks(x)
ax.set_xticklabels(["mean-shift", "global-shift", "identity", "cVAE"],
                   rotation=20)
ax.set_ylabel("per-cell Pearson r")
ax.set_ylim(0.90, 0.96)
ax.set_title("c. Held-out per-cell: simple estimators win")
ax.legend(fontsize=7.5, loc="lower left")

# d) 去噪增益
ax = axes[1, 0]
labels = ["200-400", "400-700", "700-1500", ">1500"]
ax.bar(range(4), den["mse_gain"] * 100, color="#b47cc7")
for i, v in enumerate(den["mse_gain"] * 100):
    ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(labels)
ax.set_xlabel("cells per condition")
ax.set_ylabel("MSE reduction vs sample mean (%)")
ax.set_title("d. Delta denoising: structure pays off (n=195)")

# e) 组合预测
ax = axes[1, 1]
names = ["MAGWorld-add", "MeanDelta-add", "ObsAdditivity"]
v = ndb.set_index("model").reindex(names)
x = np.arange(3)
ax.bar(x, v["delta_pearson"], 0.5,
       color=["#4878d0", "#8c8c8c", "#d65f5f"])
for i, val in enumerate(v["delta_pearson"]):
    ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(["model\n(predicted\nsingles)", "mean\nadditive",
                    "observed-data\nadditivity\n(reference)"], fontsize=8)
ax.set_ylabel("combo delta Pearson r")
ax.set_title("e. Unseen double-gene combos (n=131)")

# f) 协同
ax = axes[1, 2]
bp = ax.boxplot([syn["synergy_obs"].dropna(), [0.0] * len(syn)],
                tick_labels=["observed\ncombos", "additive\nmodel"],
                patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], ["#4878d0", "#d65f5f"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("synergy score")
ax.set_title(f"f. Synergy: observed mean={syn.synergy_obs.mean():.2f},\n"
             "additive model = 0 by construction")

plt.tight_layout()
plt.savefig(f"{OUT}/Fig6_benchmark_v2.png", dpi=170, bbox_inches="tight")
print("saved", f"{OUT}/Fig6_benchmark_v2.png")
print("FIG6V2_DONE")
