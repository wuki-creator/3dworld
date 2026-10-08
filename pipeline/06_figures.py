# -*- coding: utf-8 -*-
"""MAGWorld 论文图版组装（Nature Communications 风格多面板图）"""
import os
import glob
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

CORE = "/nfs_beijing/zizhuo/vcc/results/core"
VEL = "/nfs_beijing/zizhuo/vcc/results/velocity"
FUN = "/nfs_beijing/zizhuo/vcc/results/function"
CCC = "/nfs_beijing/zizhuo/vcc/results/ccc"
MW = "/nfs_beijing/zizhuo/vcc/results/magworld"
OUT = "/nfs_beijing/zizhuo/vcc/results/figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def fig1_overview():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.4, color="#333"))

    box(0.02, 0.62, 0.17, 0.3, "Multi-dataset\nsingle-cell atlases\n"
        "(6 public datasets)", "#dbe9f6")
    box(0.02, 0.12, 0.17, 0.3, "Perturb-seq\n(Xie/Datlinger CRISPRi\n"
        "+ drug combo)", "#f6e2db")
    box(0.28, 0.62, 0.2, 0.3, "World VAE\nuniversal cell-state\n"
        "representation", "#e2f0d9")
    box(0.28, 0.12, 0.2, 0.3, "Gene-embedding\nperturbation\noperator",
        "#e2f0d9")
    box(0.56, 0.37, 0.2, 0.3, "MAGWorld\nvirtual cell\nworld model",
        "#fff2cc")
    box(0.82, 0.62, 0.16, 0.3, "In silico\nKO / combo\n/ dose", "#ead1f7")
    box(0.82, 0.12, 0.16, 0.3, "Benchmark\n(LOO perturb.)", "#ead1f7")
    arrow(0.19, 0.77, 0.28, 0.77)
    arrow(0.19, 0.27, 0.28, 0.27)
    arrow(0.48, 0.72, 0.56, 0.58)
    arrow(0.48, 0.27, 0.56, 0.45)
    arrow(0.76, 0.55, 0.82, 0.72)
    arrow(0.76, 0.5, 0.82, 0.32)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("MAGWorld: an agent-guided world model of the virtual cell",
                 fontsize=12, weight="bold")
    save(fig, "Fig1_overview")


def fig2_atlas():
    order = [("pbmc3k", "leiden"), ("xiehon", "leiden"), ("dat17", "leiden"),
             ("aissa", "leiden"), ("pancreas", "leiden"),
             ("dentategyrus", "leiden")]
    titles = {"pbmc3k": "PBMC 3k (human)", "xiehon": "Xie CRISPRi (K562)",
              "dat17": "Datlinger CRISPRi (K562)", "aissa": "Aissa drugs (lung)",
              "pancreas": "Pancreas (mouse)", "dentategyrus": "Dentate gyrus (mouse)"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    n_cells = {}
    for i, (name, color) in enumerate(order):
        ad = sc.read_h5ad(f"{CORE}/{name}_processed.h5ad")
        n_cells[name] = int(ad.n_obs)
        sc.pl.umap(ad, color=color, ax=axes[i], show=False,
                   legend_loc="on data", legend_fontsize=6, frameon=False,
                   title=f"{titles[name]}\n({ad.n_obs:,} cells)")
    axes[-1].axis("off")
    fig.suptitle("Single-cell atlases across six public datasets",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "Fig2_atlas")
    with open(f"{OUT}/n_cells.json", "w") as f:
        json.dump(n_cells, f)


def fig3_velocity():
    scv = __import__("scvelo")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for r, name in enumerate(["pancreas", "dentategyrus"]):
        ad = sc.read_h5ad(f"{VEL}/{name}_velocity.h5ad")
        scv.pl.velocity_embedding_stream(ad, basis="umap", color="leiden",
                                         ax=axes[r][0], show=False,
                                         title=f"{name}: RNA velocity",
                                         legend_fontsize=6, frameon=False)
        scv.pl.scatter(ad, color="latent_time", color_map="gnuplot",
                       ax=axes[r][1], show=False,
                       title=f"{name}: latent time", frameon=False)
    fig.suptitle("Dynamical RNA velocity and latent developmental time",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "Fig3_velocity")


def fig4_function():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    tf = pd.read_csv(f"{FUN}/aissa_tf_by_perturbation.csv", index_col=0)
    var = tf.var(axis=1).sort_values(ascending=False)
    top = tf.loc[var.index[:20]]
    vmax1 = np.percentile(np.abs(top.values), 95) + 1e-6
    im = axes[0].imshow(top.values, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax1, vmax=vmax1)
    axes[0].set_xticks(range(len(top.columns)))
    axes[0].set_xticklabels(top.columns, rotation=90, fontsize=6)
    axes[0].set_yticks(range(len(top.index)))
    axes[0].set_yticklabels(top.index, fontsize=6)
    axes[0].set_title("TF activity, Aissa (CollecTRI)", fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.7)
    pwf = f"{FUN}/pbmc3k_pathway_by_leiden.csv"
    if not os.path.exists(pwf):
        pwf = f"{FUN}/dat17_tf_by_perturbation.csv"
    pw = pd.read_csv(pwf, index_col=0)
    if pw.shape[0] > 25:
        pw = pw.loc[pw.var(axis=1).sort_values(ascending=False).index[:25]]
    im2 = axes[1].imshow(pw.values, aspect="auto", cmap="viridis")
    axes[1].set_xticks(range(len(pw.columns)))
    axes[1].set_xticklabels(pw.columns, rotation=0, fontsize=7)
    axes[1].set_yticks(range(len(pw.index)))
    axes[1].set_yticklabels(pw.index, fontsize=7)
    ttl = ("Pathway activity, PBMC3k (PROGENy)" if "pathway" in pwf
           else "TF activity, Datlinger (CollecTRI)")
    axes[1].set_title(ttl, fontsize=10)
    fig.colorbar(im2, ax=axes[1], shrink=0.7)
    gs = sorted(glob.glob(f"{FUN}/pbmc3k_gsea_cluster*.csv"))
    if gs:
        g = pd.read_csv(gs[0]).head(8)[::-1]
        axes[2].barh(g["Term"], -np.log10(
            g["FDR q-val"].astype(float).clip(1e-10)), color="#4878cf")
        axes[2].set_xlabel("-log10(FDR)")
        cl = os.path.basename(gs[0]).replace("pbmc3k_gsea_cluster", "")[:-4]
        axes[2].set_title(f"GSEA, PBMC3k cluster {cl}", fontsize=10)
        axes[2].tick_params(axis="y", labelsize=7)
    fig.suptitle("Functional downstream analyses", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "Fig4_function")


def fig5_ccc():
    import matplotlib.image as mpimg
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ed = pd.read_csv(f"{CCC}/pbmc3k_ccc_edges.csv")
    top = ed.head(15)
    axes[0].barh(range(len(top))[::-1], np.log10(top["score"] + 1),
                 color=plt.cm.cividis(np.linspace(0.2, 0.9, len(top))))
    axes[0].set_yticks(range(len(top))[::-1])
    axes[0].set_yticklabels((top["ligand"].astype(str) + "\u2192" +
                             top["receptor"].astype(str) + "  " +
                             top["source"].astype(str) + "\u21d2" +
                             top["target"].astype(str)).tolist(),
                            fontsize=6)
    axes[0].set_xlabel("log10(1+score)")
    axes[0].set_title("PBMC3k: top ligand-receptor interactions", fontsize=10)
    heat = mpimg.imread(f"{CCC}/pbmc3k_ccc_heatmap.png")
    axes[1].imshow(heat)
    axes[1].axis("off")
    axes[1].set_title("Cluster-to-cluster communication", fontsize=10)
    fig.suptitle("Cell-cell communication landscape", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "Fig5_ccc")


def fig6_magworld():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    res = pd.read_csv(f"{MW}/loo_benchmark.csv")
    # Identity 的效应相关无定义 (零向量), 不参与箱线图
    models = ["MAGWorld", "MeanDelta", "PermutedNull"]
    colors = {"MAGWorld": "#d1495b", "MeanDelta": "#4878cf",
              "Identity": "#b7b7b7", "PermutedNull": "#cccccc"}
    for k, (metric, ylab, ttl) in enumerate([
            ("delta_pearson", "Effect Pearson r (delta)",
             "a. Perturbation effect prediction"),
            ("deg_top50_overlap", "DEG top-50 overlap", "b. DEG recovery")]):
        data = [res[res.model == m][metric].dropna().values for m in models]
        try:
            bp = axes[0][k].boxplot(data, tick_labels=models,
                                    patch_artist=True)
        except TypeError:  # matplotlib < 3.9
            bp = axes[0][k].boxplot(data, labels=models, patch_artist=True)
        for p, m in zip(bp["boxes"], models):
            p.set_facecolor(colors[m])
        axes[0][k].set_ylabel(ylab)
        axes[0][k].set_title(ttl, fontsize=10)
        axes[0][k].set_xticks(range(1, len(models) + 1))
        axes[0][k].set_xticklabels(models, rotation=15, fontsize=8)
    disc = pd.read_csv(f"{MW}/perturbation_discrimination.csv")
    axes[0][2].bar(range(len(disc)), disc["cosine_rank"], color="#4878cf")
    axes[0][2].axhline(1, color="#d1495b", ls="--", lw=1)
    axes[0][2].set_xticks(range(len(disc)))
    axes[0][2].set_xticklabels(disc["perturbation"], rotation=90, fontsize=6)
    axes[0][2].set_ylabel("Cosine rank of true effect")
    axes[0][2].set_title("c. Perturbation discrimination", fontsize=10)
    # 归一化 delta MAE bar
    summ = res.groupby("model")["delta_mae_norm"].mean().reindex(models)
    axes[1][0].bar(models, summ.values, color=[colors[m] for m in models])
    axes[1][0].set_ylabel("Normalized delta MAE")
    axes[1][0].set_title("d. Effect magnitude error", fontsize=10)
    axes[1][0].set_xticks(range(len(models)))
    axes[1][0].set_xticklabels(models, rotation=15, fontsize=8)
    # 剂量反应 (画相对对照的偏移量)
    dz = np.load(f"{MW}/dose_response.npz")
    dose = dz["dose"]
    with open(f"{MW}/dose_response_perts.json") as f:
        perts = json.load(f)
    cmap = plt.cm.tab10
    for i, k in enumerate([k for k in dz.files if k.startswith("curve_")]):
        idx = int(k.split("_")[1])
        dev = dz[k] - dz[k][0]
        gidx = np.argsort(dev[-1] - dev[0])[-3:]
        for j, gi in enumerate(gidx):
            axes[1][1].plot(dose, dev[:, gi], marker="o", ms=3,
                            color=cmap(j), alpha=0.5 if i else 0.9)
    axes[1][1].axhline(0, color="#333", lw=0.6, ls=":")
    axes[1][1].set_xlabel("Perturbation intensity")
    axes[1][1].set_ylabel("Deviation from control")
    axes[1][1].set_title("e. In silico dose response", fontsize=10)
    # 药物组合: 双药效应预测 vs 加性模型
    combo = pd.read_csv(f"{MW}/drug_combo_benchmark.csv")
    cc = combo.set_index("model").reindex(
        ["MAGWorld-combo", "Additivity", "Identity"])
    axes[1][2].bar(range(len(cc)), cc["delta_pearson"].values,
                   color=["#d1495b", "#4878cf", "#b7b7b7"][:len(cc)])
    axes[1][2].set_xticks(range(len(cc)))
    axes[1][2].set_xticklabels(cc.index, rotation=15, fontsize=8)
    axes[1][2].set_ylabel("Effect Pearson r (delta)")
    axes[1][2].set_title("f. Osimertinib+crizotinib combo", fontsize=10)
    fig.suptitle("MAGWorld virtual-cell benchmark on Datlinger CRISPRi perturb-seq",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "Fig6_magworld")


def fig7_qc():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, name in enumerate(["pbmc3k", "aissa", "pancreas"]):
        ad = sc.read_h5ad(f"{CORE}/{name}_processed.h5ad")
        sc.pl.violin(ad, ["n_genes_by_counts", "pct_counts_mt"],
                     ax=axes[0][i], show=False, jitter=0.2, stripplot=False)
        axes[0][i].set_title(f"{name}", fontsize=9)
        sc.pl.scatter(ad, x="n_genes_by_counts", y="pct_counts_mt",
                      ax=axes[1][i], show=False)
        axes[1][i].set_title(f"{name}", fontsize=9)
    fig.suptitle("Quality control across datasets", fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "Fig7_qc")


def fig8_ko():
    act = pd.read_csv(f"{MW}/insilico_ko_tf_activity.csv")
    summ = pd.read_csv(f"{MW}/insilico_ko_summary.csv")
    ko_list = list(summ["ko_target"])
    # 固定 TF 集合: 所有 KO 的自身靶 TF + 变化最大的并集 (保证行/列对齐)
    sub_all = act[act.ko_target.isin(ko_list)]
    top_by_var = sub_all.groupby("tf")["delta_activity"].apply(
        lambda s: s.abs().max()).sort_values(ascending=False)
    keep = [t for t in top_by_var.index[:10]] + \
           [t for t in ko_list if t not in top_by_var.index[:10]]
    keep = keep[:14]
    mat = []
    for ko in ko_list:
        sub = act[act.ko_target == ko].set_index("tf").reindex(keep)
        mat.append(sub["delta_activity"].fillna(0).values)
    M = np.vstack(mat)
    vmax = max(abs(M).max(), 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    im = axes[0].imshow(M, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
    axes[0].set_yticks(range(len(ko_list)))
    axes[0].set_yticklabels([f"KO {k}" for k in ko_list], fontsize=8)
    axes[0].set_xticks(range(len(keep)))
    axes[0].set_xticklabels(
        [t + ("*" if t in ko_list else "") for t in keep],
        rotation=90, fontsize=7)
    for i, ko in enumerate(ko_list):
        if ko in keep:
            j = keep.index(ko)
            axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, ec="k", lw=1.5))
    axes[0].set_title("a. TF activity change after in silico KO "
                      "(* = knocked-out TF)", fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.7, label="delta TF activity (ULM)")
    axes[1].bar(range(len(summ)), summ["self_tf_delta"],
                color=["#d1495b" if v < 0 else "#4878cf"
                       for v in summ["self_tf_delta"]])
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_xticks(range(len(summ)))
    axes[1].set_xticklabels(summ["ko_target"], rotation=30, fontsize=8)
    axes[1].set_ylabel("Own TF activity change")
    axes[1].set_title("b. Self-regulation consistency", fontsize=10)
    fig.suptitle("In silico transcription-factor knockout in T-cell virtual cells "
                 "(Datlinger CRISPRi control manifold)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    save(fig, "Fig8_ko")


if __name__ == "__main__":
    fig1_overview()
    fig2_atlas()
    for fn in (fig3_velocity, fig4_function, fig5_ccc, fig6_magworld,
               fig7_qc, fig8_ko):
        try:
            fn()
        except Exception as e:
            print(fn.__name__, "skip:", str(e)[:200])
    print("ALL_FIGURES_DONE")
