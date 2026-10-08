# -*- coding: utf-8 -*-
"""Build gene-family (paralog proxy) features from HGNC complete set.

Outputs: hgnc_family_features.csv  (symbol, family_size, n_groups)
family_size = number of approved genes sharing any gene_group_id with the gene
              (the gene itself included); 1 for genes in no group.
"""
import csv

groups = {}  # group_id -> set(symbols)
sym_groups = {}  # symbol -> set(group_ids)

with open("hgnc_complete_set.txt", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    i_sym = header.index("symbol")
    i_gid = header.index("gene_group_id")
    for row in rd:
        if len(row) <= max(i_sym, i_gid):
            continue
        sym = row[i_sym].strip()
        if not sym:
            continue
        ids = [g.strip() for g in row[i_gid].split("|") if g.strip()]
        if not ids:
            continue
        sym_groups.setdefault(sym, set()).update(ids)
        for g in ids:
            groups.setdefault(g, set()).add(sym)

with open("hgnc_family_features.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["symbol", "family_size", "n_groups"])
    for sym, gids in sym_groups.items():
        fam = set()
        for g in gids:
            fam |= groups[g]
        w.writerow([sym, len(fam), len(gids)])

# stats
import collections
sizes = [len(set().union(*(groups[g] for g in gids))) for gids in sym_groups.values()]
print("genes with family info:", len(sizes))
print("median family size:", sorted(sizes)[len(sizes)//2])
print("max family size:", max(sizes))
