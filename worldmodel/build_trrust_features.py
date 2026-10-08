# -*- coding: utf-8 -*-
"""Build TRRUST prior: 6-dim count features + directed signed graph, panel-aligned.
Outputs: trrust_counts6.npy (18533x6), trrust_graph.npz (edge_index/sign/w)
"""
import csv
import numpy as np

# panel
with open("gene_names.csv") as fh:
    lines = [l.strip().strip('"').strip(",") for l in fh]
panel = [g for g in lines if g and g != "gene_name"]
N = len(panel); idx = {g: i for i, g in enumerate(panel)}

# HGNC symbol normalization
approved = set(); syn2sym = {}
with open("hgnc_complete_set.txt", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    i_sym = header.index("symbol"); i_prev = header.index("prev_symbol"); i_alias = header.index("alias_symbol")
    for row in rd:
        if len(row) <= max(i_sym, i_prev, i_alias): continue
        sym = row[i_sym].strip()
        if not sym: continue
        approved.add(sym)
        for o in (row[i_prev] + "|" + row[i_alias]).split("|"):
            o = o.strip()
            if o: syn2sym[o.upper()] = sym
def canon(g):
    g = g.strip().upper()
    return g if g in approved else syn2sym.get(g, g)

SIGN = {"Activation": 1, "Repression": -1}
cnt = np.zeros((N, 6), dtype=np.float32)  # out_act out_rep out_unk in_act in_rep in_unk
edges = {}  # (src,dst) -> [sign_sum, weight]
with open("trrust_rawdata.human.tsv") as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) < 3: continue
        tf, tg, mode = canon(p[0]), canon(p[1]), p[2]
        if tf not in idx or tg not in idx: continue
        s = SIGN.get(mode, 0)
        i, j = idx[tf], idx[tg]
        col = {1: 0, -1: 1}.get(s, 2)
        cnt[i, col] += 1          # out
        cnt[j, 3 + col] += 1      # in
        if s != 0:
            k = (i, j)
            if k not in edges:
                edges[k] = [s, 0]
            edges[k][1] += 1      # weight = #supporting rows

np.save("trrust_counts6.npy", cnt)
ei = np.array([[k[0] for k in edges], [k[1] for k in edges]], dtype=np.int32)
sg = np.array([v[0] for v in edges.values()], dtype=np.float32)
w  = np.array([v[1] for v in edges.values()], dtype=np.float32)
np.savez("trrust_graph.npz", edge_index=ei, edge_sign=sg, edge_w=w)
print("counts:", cnt.shape, "out_act genes:", int((cnt[:,0]>0).sum()), "in edges targets:", int((cnt[:,3:].sum(1)>0).sum()))
print("graph edges:", ei.shape[1], "act:", int((sg>0).sum()), "rep:", int((sg<0).sum()))

# also build 14-dim combined counts: dgidb8 + trrust6
dg = np.load("dgidb_prior_features.npy").astype(np.float32)
assert dg.shape[0] == N
comb = np.concatenate([dg, cnt], axis=1)
np.save("prior_counts14.npy", comb)
print("combined:", comb.shape)
