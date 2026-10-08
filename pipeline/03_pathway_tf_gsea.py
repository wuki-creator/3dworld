# -*- coding: utf-8 -*-
"""MAGWorld 下游功能分析 v2（本地 CollecTRI 资源，不再依赖外网）
1) TF 活性 (CollecTRI + dc.mt.ulm)
2) 通路活性 (PROGENy, 外网不稳 -> try/except 跳过)
3) GSEA 聚类标志基因 (pbmc3k)
4) GSEA 扰动 vs control (aissa 药物, dat17/xiehon CRISPRi 可选)
输出: results/function/
"""
import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

CORE = "/nfs_beijing/zizhuo/vcc/results/core"
OUT = "/nfs_beijing/zizhuo/vcc/results/function"
RES = "/nfs_beijing/zizhuo/vcc/data/resources"
os.makedirs(OUT, exist_ok=True)


def load_collectri():
    net = pd.read_csv(f"{RES}/CollecTRI_regulons.csv")
    net = net[["source", "target", "weight"]].dropna()
    print(f"CollecTRI local: {net.shape[0]} edges, {net.source.nunique()} TFs",
          flush=True)
    return net


def _run(mat_adata, net, method, name, tag):
    fn = getattr(dc.mt, method)
    before = set(mat_adata.obsm.keys())
    fn(mat_adata, net, verbose=False)
    new = [k for k in mat_adata.obsm.keys()
           if k not in before and str(k).startswith("score")]
    if not new:
        raise RuntimeError(f"no score key produced by {method}")
    est = pd.DataFrame(mat_adata.obsm[new[0]])
    est.index = mat_adata.obs_names
    est.to_csv(f"{OUT}/{name}_{tag}_acts.csv")
    return est


def tf_and_pathway(name, ad, groupby, net_t):
    tf = _run(ad, net_t, "ulm", name, "tf")
    try:
        net_p = dc.op.progeny(organism="human")
        pw = _run(ad, net_p, "mlm", name, "pathway")
    except Exception as e:
        print(f"[{name}] progeny skipped: {str(e)[:80]}", flush=True)
        pw = None
    groups = ad.obs[groupby].astype(str)
    tf.groupby(groups).mean().to_csv(f"{OUT}/{name}_tf_by_{groupby}.csv")
    if pw is not None:
        pw.groupby(groups).mean().to_csv(
            f"{OUT}/{name}_pathway_by_{groupby}.csv")
    print(f"[{name}] TF/pathway DONE", flush=True)


def gsea_clusters(ad, tag, n_clusters=8):
    mk = sc.get.rank_genes_groups_df(ad, None, key="markers")
    import gseapy as gp
    for cl in mk["group"].unique()[:n_clusters]:
        degs = mk[mk["group"] == cl].sort_values("scores", ascending=False)
        rnk = pd.Series(degs["scores"].values, index=degs["names"].values)
        try:
            enr = gp.prerank(rnk=rnk, gene_sets="KEGG_2021_Human",
                             permutation_num=200, no_plot=True, seed=0,
                             verbose=False)
            enr.res2d.sort_values("FDR q-val").head(10).to_csv(
                f"{OUT}/{tag}_gsea_cluster{cl}.csv", index=False)
        except Exception as e:
            print("gsea fail", cl, str(e)[:80])
    print(f"[{tag}] GSEA DONE", flush=True)


def gsea_perturbation(ad, name, groupby, control_label, max_perts=8):
    """每个扰动 vs control 的差异基因做 prerank GSEA"""
    import gseapy as gp
    obs = ad.obs[groupby].astype(str)
    perts = [p for p in obs.value_counts().index
             if p != control_label][:max_perts]
    sc.tl.rank_genes_groups(ad, groupby, groups=perts,
                            reference=control_label, method="wilcoxon",
                            key_added="deg_vs_ctrl")
    mk = sc.get.rank_genes_groups_df(ad, None, key="deg_vs_ctrl")
    for p in perts:
        degs = mk[mk["group"] == p].dropna(subset=["scores"])
        degs = degs.sort_values("scores", ascending=False)
        rnk = pd.Series(degs["scores"].values, index=degs["names"].values)
        rnk = rnk[~rnk.index.duplicated()]
        try:
            enr = gp.prerank(rnk=rnk, gene_sets="KEGG_2021_Human",
                             permutation_num=200, no_plot=True, seed=0,
                             verbose=False)
            enr.res2d.sort_values("FDR q-val").head(10).to_csv(
                f"{OUT}/{name}_gsea_{p}.csv", index=False)
            print(f"[{name}] gsea {p} DONE", flush=True)
        except Exception as e:
            print(f"gsea fail {p}", str(e)[:80])


def main():
    net_t = load_collectri()

    pb = sc.read_h5ad(f"{CORE}/pbmc3k_processed.h5ad")
    tf_and_pathway("pbmc3k", pb, "leiden", net_t)
    gsea_clusters(pb, "pbmc3k")

    ai = sc.read_h5ad(f"{CORE}/aissa_processed.h5ad")
    pc = ai.obs["perturbation"].astype(str).value_counts()
    keep = pc[pc >= 100].index[:20]
    ai_s = ai[ai.obs["perturbation"].astype(str).isin(keep)].copy()
    tf_and_pathway("aissa", ai_s, "perturbation", net_t)
    ctrl = "control" if "control" in ai.obs["perturbation"].astype(
        str).unique() else ai.obs["perturbation"].astype(str).value_counts().idxmax()
    gsea_perturbation(ai_s, "aissa", "perturbation", ctrl)

    dt = sc.read_h5ad(f"{CORE}/dat17_processed.h5ad")
    if "perturbation" in dt.obs.columns:
        pc = dt.obs["perturbation"].astype(str).value_counts()
        keep = pc[pc >= 100].index[:20]
        dt_s = dt[dt.obs["perturbation"].astype(str).isin(keep)].copy()
        tf_and_pathway("dat17", dt_s, "perturbation", net_t)

    xh = f"{CORE}/xiehon_processed.h5ad"
    if os.path.exists(xh):
        xi = sc.read_h5ad(xh)
        if "perturbation" in xi.obs.columns:
            pc = xi.obs["perturbation"].astype(str).value_counts()
            keep = pc[pc >= 100].index[:20]
            if len(keep) >= 2:
                xi_s = xi[xi.obs["perturbation"].astype(str).isin(keep)].copy()
                tf_and_pathway("xiehon", xi_s, "perturbation", net_t)
    print("ALL_FUNCTION_DONE")


if __name__ == "__main__":
    main()
