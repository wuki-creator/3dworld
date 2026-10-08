# -*- coding: utf-8 -*-
"""MAGWorld 细胞-细胞通讯分析（decoupler 2.x: dc.op.resource OmniPath 配体-受体）
pbmc3k（人免疫）与 dat17（K562 CRISPRi 按聚类）的簇间通讯网络。
"""
import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
CORE = "/nfs_beijing/zizhuo/vcc/results/core"
OUT = "/nfs_beijing/zizhuo/vcc/results/ccc"
os.makedirs(OUT, exist_ok=True)


BUILTIN_LR = [
    ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"), ("IL6", "IL6R"), ("IL6", "IL6ST"),
    ("TNF", "TNFRSF1A"), ("TNF", "TNFRSF1B"), ("IFNG", "IFNGR1"), ("IFNG", "IFNGR2"),
    ("IL1B", "IL1R1"), ("IL1B", "IL1RAP"), ("CSF1", "CSF1R"), ("CSF2", "CSF2RA"),
    ("IL10", "IL10RA"), ("IL10", "IL10RB"), ("CCL2", "CCR2"), ("CCL5", "CCR5"),
    ("CXCL12", "CXCR4"), ("CXCL8", "CXCR1"), ("CXCL8", "CXCR2"), ("VEGFA", "KDR"),
    ("VEGFA", "FLT1"), ("EGF", "EGFR"), ("PDGFB", "PDGFRB"), ("IGF1", "IGF1R"),
    ("HGF", "MET"), ("FGF2", "FGFR1"), ("KITLG", "KIT"), ("APP", "CD74"),
    ("HLA-DRA", "CD4"), ("CALR", "LRP1"), ("THBS1", "CD47"), ("THBS1", "CD36"),
    ("CD74", "APP"), ("SAA1", "TLR4"), ("HMGB1", "TLR4"), ("HMGB1", "AGER"),
    ("LTA", "LTBR"), ("EDA", "EDAR"), ("TNFSF10", "TNFRSF10A"), ("TNFSF10", "TNFRSF10B"),
    ("SEMA3C", "NRP1"), ("SEMA3C", "NRP2"), ("PSAP", "GPR37"),
]


def get_lr():
    try:
        op = dc.op.resource("OmniPath")
        if "category" in op.columns:
            op = op[op["category"] == "ligand-receptor"]
        src = "source_genesymbol" if "source_genesymbol" in op.columns else "source"
        tgt = "target_genesymbol" if "target_genesymbol" in op.columns else "target"
        lr = op[[src, tgt]].dropna().drop_duplicates()
        lr.columns = ["ligand", "receptor"]
        print(f"LR source: OmniPath ({len(lr)} pairs)", flush=True)
        return lr
    except Exception as e:
        print("OmniPath unavailable:", str(e)[:100], flush=True)
    try:
        import urllib.request
        url = ("https://raw.githubusercontent.com/ventolab/CellphoneDB-data/"
               "master/data/interaction_input.csv")
        csv = urllib.request.urlopen(url, timeout=60).read()
        import io as _io
        cp = pd.read_csv(_io.BytesIO(csv))
        gene_a = [c for c in cp.columns if "gene_a" in c.lower()][0]
        gene_b = [c for c in cp.columns if "gene_b" in c.lower()][0]
        lr = cp[[gene_a, gene_b]].dropna().drop_duplicates()
        lr.columns = ["ligand", "receptor"]
        print(f"LR source: CellPhoneDB ({len(lr)} pairs)", flush=True)
        return lr
    except Exception as e:
        print("CellPhoneDB unavailable:", str(e)[:100], flush=True)
    lr = pd.DataFrame(BUILTIN_LR, columns=["ligand", "receptor"])
    print(f"LR source: builtin ({len(lr)} pairs)", flush=True)
    return lr


def ccc(name, ad, groupby, top_n=25):
    lr = get_lr()
    X = pd.DataFrame(ad.raw.X.toarray() if hasattr(ad.raw.X, "toarray")
                     else ad.raw.X, index=ad.obs_names,
                     columns=ad.raw.var_names)
    expr = X.groupby(ad.obs[groupby].astype(str)).mean()
    genes = set(expr.columns)
    lr = lr[lr["ligand"].isin(genes) & lr["receptor"].isin(genes)]
    print(f"[{name}] LR pairs retained: {len(lr)}", flush=True)
    groups = list(expr.index)
    rows = []
    for l, r in lr.itertuples(index=False):
        lev, rev = expr[l], expr[r]
        for a in groups:
            la = lev[a]
            if la <= 0:
                continue
            for b in groups:
                if a != b and rev[b] > 0:
                    rows.append((a, b, l, r, la * rev[b]))
    ed = pd.DataFrame(rows, columns=["source", "target", "ligand",
                                     "receptor", "score"])
    ed = ed.sort_values("score", ascending=False)
    ed.to_csv(f"{OUT}/{name}_ccc_edges.csv", index=False)

    top = ed.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = (top["ligand"] + "->" + top["receptor"] + "\n" +
              top["source"] + "=>" + top["target"])
    ax.barh(range(len(top))[::-1], top["score"],
            color=plt.cm.viridis(np.linspace(0.2, 0.9, len(top))))
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("LR expression product")
    ax.set_title(f"{name}: top {top_n} CCC interactions")
    plt.tight_layout()
    fig.savefig(f"{OUT}/{name}_ccc_top.png", dpi=200)
    plt.close(fig)

    mat = ed.groupby(["source", "target"])["score"].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log1p(mat.values), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title(f"{name}: cluster communication (log score)")
    fig.colorbar(im, label="log(1+sum)")
    plt.tight_layout()
    fig.savefig(f"{OUT}/{name}_ccc_heatmap.png", dpi=200)
    plt.close(fig)
    print(f"[{name}] CCC DONE edges={len(ed)}", flush=True)


if __name__ == "__main__":
    pb = sc.read_h5ad(f"{CORE}/pbmc3k_processed.h5ad")
    ccc("pbmc3k", pb, "leiden")
    dt = sc.read_h5ad(f"{CORE}/dat17_processed.h5ad")
    ccc("dat17", dt, "leiden")
    print("ALL_CCC_DONE")
