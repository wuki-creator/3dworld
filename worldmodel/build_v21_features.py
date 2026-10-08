# -*- coding: utf-8 -*-
"""Build v21 prior features: DGIDB(8) + essentiality(2) + HGNC family(2) = 12 dims.
Output: prior_v21.npy aligned to local gene_names.csv panel order.
"""
import csv
import numpy as np

# ---- panel genes ----
genes = []
with open("gene_names.csv") as fh:
    lines = [l.strip().strip('"').strip(",") for l in fh]
genes = [g for g in lines if g and g != "gene_name"]
print("panel genes:", len(genes), genes[:3])
panel = list(dict.fromkeys(genes))
N = len(panel)
idx = {g: i for i, g in enumerate(panel)}

# ---- HGNC symbol normalization (prev/alias -> approved) ----
approved = set()
syn2sym = {}
with open("hgnc_complete_set.txt", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    i_sym = header.index("symbol")
    i_prev = header.index("prev_symbol")
    i_alias = header.index("alias_symbol")
    i_gid = header.index("gene_group_id")
    fam_size = {}
    n_groups = {}
    for row in rd:
        if len(row) <= max(i_sym, i_prev, i_alias, i_gid):
            continue
        sym = row[i_sym].strip()
        if not sym:
            continue
        approved.add(sym)
        for other in (row[i_prev] + "|" + row[i_alias]).split("|"):
            o = other.strip()
            if o:
                syn2sym[o.upper()] = sym
        ids = [g.strip() for g in row[i_gid].split("|") if g.strip()]
        n_groups[sym] = len(ids)

def canon(g):
    g = g.strip().upper()
    return g if g in approved else syn2sym.get(g, g)

# family size: count approved genes sharing a group — recompute per panel gene
groups = {}
with open("hgnc_complete_set.txt", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    i_sym = header.index("symbol")
    i_gid = header.index("gene_group_id")
    for row in rd:
        if len(row) <= max(i_sym, i_gid):
            continue
        sym = row[i_sym].strip()
        for g in row[i_gid].split("|"):
            g = g.strip()
            if g and sym:
                groups.setdefault(g, set()).add(sym)

# ---- essentiality lists ----
import pyreadr
ess = set()
for f in ["ADaM2021_essential.RData", "BAGEL_essential_v2.RData"]:
    d = pyreadr.read_r(f)
    for k, v in d.items():
        for g in v.iloc[:, 0]:
            ess.add(canon(str(g)))
noness = set()
d = pyreadr.read_r("BAGEL_nonEssential.RData")
for k, v in d.items():
    for g in v.iloc[:, 0]:
        noness.add(canon(str(g)))
noness -= ess
print("essential mapped:", len(ess), "nonessential:", len(noness))

# ---- assemble 12-dim features ----
dg = np.load("dgidb_prior_features.npy").astype(np.float32)
assert dg.shape[0] == N, (dg.shape, N)

feats = np.zeros((N, 12), dtype=np.float32)
feats[:, :8] = dg
c_ess = c_non = 0
for i, g in enumerate(panel):
    cg = canon(g)
    if cg in ess:
        feats[i, 8] = 1.0; c_ess += 1
    if cg in noness:
        feats[i, 9] = 1.0; c_non += 1
    fs = 0
    # family size via groups of this gene: need its group ids
    feats[i, 10] = 0.0
# second pass for family: build sym->group-ids map
sym2gids = {}
with open("hgnc_complete_set.txt", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    i_sym = header.index("symbol")
    i_gid = header.index("gene_group_id")
    for row in rd:
        if len(row) <= max(i_sym, i_gid):
            continue
        ids = [g.strip() for g in row[i_gid].split("|") if g.strip()]
        if row[i_sym].strip():
            sym2gids[row[i_sym].strip()] = ids
for i, g in enumerate(panel):
    cg = canon(g)
    ids = sym2gids.get(cg, [])
    fam = set()
    for x in ids:
        fam |= groups.get(x, set())
    feats[i, 10] = np.log1p(len(fam)) if fam else 0.0
    feats[i, 11] = np.log1p(len(ids))

print("panel essential hits:", c_ess, "nonessential hits:", c_non)
print("family feats mean:", feats[:,10].mean(), "ngroups mean:", feats[:,11].mean())
np.save("prior_v21.npy", feats)
print("saved prior_v21.npy", feats.shape)
