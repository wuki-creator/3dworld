# -*- coding: utf-8 -*-
"""Build Norman prep spec locally (genes/barcodes/identities already local).
Output: norman_spec.npz
  gpanel   (G_file,) int32   panel index per matrix gene row (-1 = unmapped)
  ctrl_rows    (C,) int32   matrix cell rows for control cells
  cond_names   (M,) str     perturbed gene symbol per condition
  cond_rows    object (M,) of int32 arrays (matrix cell rows per condition)
"""
import csv
import numpy as np
import pandas as pd

# ---- panel ----
with open("gene_names.csv") as fh:
    lines = [l.strip().strip('"').strip(",") for l in fh]
panel = [g for g in lines if g and g != "gene_name"]
pidx = {g: i for i, g in enumerate(panel)}

# ---- HGNC canon ----
approved = set(); syn2sym = {}
with open("hgnc_complete_set.txt", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    i_sym = header.index("symbol"); i_prev = header.index("prev_symbol"); i_alias = header.index("alias_symbol")
    for row in rd:
        if len(row) <= max(i_sym, i_prev, i_alias):
            continue
        sym = row[i_sym].strip()
        if not sym:
            continue
        approved.add(sym)
        for o in (row[i_prev] + "|" + row[i_alias]).split("|"):
            o = o.strip()
            if o:
                syn2sym[o.upper()] = sym
def canon(g):
    g = g.strip().upper()
    return g if g in approved else syn2sym.get(g, g)

# ---- genes ----
genes = pd.read_csv("GSE133344_filtered_genes.tsv.gz", sep="\t", header=None)[1].astype(str)
gpanel = genes.map(lambda g: pidx.get(canon(g), -1)).values.astype(np.int32)
print("matrix genes:", len(genes), "mapped:", int((gpanel >= 0).sum()))

# ---- barcodes ----
bc = pd.read_csv("GSE133344_filtered_barcodes.tsv.gz", header=None)[0].astype(str)
bc_row = {b: i for i, b in enumerate(bc)}
print("matrix cells:", len(bc))

# ---- identities ----
ident = pd.read_csv("ident.csv")
ident = ident[(ident["good_coverage"] == True) & (ident["number_of_cells"] == 1)]
ident["row"] = ident["cell_barcode"].astype(str).map(bc_row)
ident = ident.dropna(subset=["row"]).copy()
ident["row"] = ident["row"].astype(int)
print("cells after QC & barcode match:", len(ident))

def parse_label(s):
    # 'KLF1_NegCtrl0__KLF1_NegCtrl0' -> combo tokens
    combo = str(s).split("__")[0]
    toks = [t for t in combo.split("_") if not t.startswith("NegCtrl")]
    if len(toks) == 0:
        return "CTRL"
    if len(set(toks)) == 1:
        return canon(toks[0])
    return None  # double perturbation

ident["cond"] = ident["guide_identity"].map(parse_label)
lab = ident["cond"].values
rows = ident["row"].values

ctrl_rows = rows[lab == "CTRL"]
print("control cells:", len(ctrl_rows))

cond_names = []
cond_rows = []
for cg in sorted(set(l for l in lab if l not in ("CTRL", None))):
    sel = rows[lab == cg]
    if cg not in pidx or len(sel) < 20:
        continue
    cond_names.append(cg)
    cond_rows.append(sel.astype(np.int32))
print("conditions:", len(cond_names), "first:", cond_names[:5])
print("total KO cells:", sum(len(r) for r in cond_rows))

np.savez("norman_spec.npz",
         gpanel=gpanel,
         ctrl_rows=ctrl_rows.astype(np.int32),
         cond_names=np.array(cond_names),
         cond_rows=np.array(cond_rows, dtype=object),
         allow_pickle=True)
print("saved norman_spec.npz")
