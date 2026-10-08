"""Build DGIDB prior feature matrix aligned to the VCC gene panel.

Features per gene (log1p weighted counts, 8 dims):
 0 inhibit_total   : inhibitor+blocker+negative modulator+antibody+inverse agonist+antagonist
 1 activate_total  : agonist+activator+positive modulator+potentiator
 2 bind_total      : binder
 3 modulate_total  : modulator (unspecified)
 4 unknown_total   : NULL/other/unknown/cleavage/vaccine/etc
 5 approved_inhibit  (approved drug only, weight 2)
 6 approved_activate (approved drug only, weight 2)
 7 log_total
Plus a signed direction prior (activate-inhibit)/total saved separately.
"""
import csv
import numpy as np

INHIBIT = {"inhibitor", "blocker", "negative modulator", "antibody",
           "inverse agonist", "antagonist", "antisense oligonucleotide"}
ACTIVATE = {"agonist", "activator", "positive modulator", "potentiator"}
BIND = {"binder"}
MODULATE = {"modulator"}

panel = []
with open("gene_names.csv", newline="", encoding="utf-8") as fh:
    rows = list(csv.reader(fh))
    if rows and "gene" in rows[0][0].lower():
        rows = rows[1:]
    panel = [r[0] for r in rows if r]
panel_set = set(panel)

feats = np.zeros((len(panel), 8), dtype=np.float64)
idx = {g: i for i, g in enumerate(panel)}

n_in = n_act = 0
with open("dgidb_interactions.tsv", newline="", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    h = {name: i for i, name in enumerate(header)}
    for row in rd:
        if len(row) < 13:
            continue
        gene = row[h["gene_name"]].strip().upper()
        if gene not in panel_set:
            continue
        itype = (row[h["interaction_type"]] or "NULL").strip().lower()
        approved = (row[h["approved"]] or "").strip().lower() in ("true", "1", "yes", "t")
        w = 2.0 if approved else 1.0
        i = idx[gene]
        if itype in INHIBIT:
            feats[i, 0] += w
            if approved:
                feats[i, 5] += w
            n_in += 1
        elif itype in ACTIVATE:
            feats[i, 1] += w
            if approved:
                feats[i, 6] += w
            n_act += 1
        elif itype in BIND:
            feats[i, 2] += w
        elif itype in MODULATE:
            feats[i, 3] += w
        else:
            feats[i, 4] += w

total = feats[:, :5].sum(1)
feats[:, 7] = np.log1p(total)
for c in range(7):
    feats[:, c] = np.log1p(feats[:, c])

direction = (feats[:, 1] - feats[:, 0]) / np.maximum(feats[:, 1] + feats[:, 0], 1.0)

np.save("dgidb_prior_features.npy", feats.astype(np.float32))
np.save("dgidb_prior_direction.npy", direction.astype(np.float32))

covered = int((total > 0).sum())
print("panel genes:", len(panel))
print("genes with any DGIDB interaction:", covered)
print("target-side (any interaction) rate: %.3f" % (covered / len(panel)))
print("inhibit rows: %d  activate rows: %d" % (n_in, n_act))
print("feature means:", feats.mean(0).round(3))
print("direction nonzero:", int((np.abs(direction) > 0).sum()))
