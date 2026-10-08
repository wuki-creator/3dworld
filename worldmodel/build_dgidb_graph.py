"""Build DGIDB gene-gene co-drug graph for GAT prior.

Edges: genes sharing a drug. Per drug with class c in {-1 (inhibit), +1 (activate)}
and k panel targets, add clique edges with weight 1/sqrt(k) on the signed
adjacency of that class. Keep top-32 neighbors per gene per sign.

Output: dgidb_graph.npz with
  edge_index: int32 (2, E) panel-gene indices
  edge_sign:  float32 (E)  +1 activate / -1 inhibit
  edge_w:     float32 (E)
"""
import csv
import numpy as np
from collections import defaultdict

panel = [r[0] for r in list(csv.reader(open("gene_names.csv", encoding="utf-8")))[1:] if r]
idx = {g: i for i, g in enumerate(panel)}

INHIBIT = {"inhibitor", "blocker", "negative modulator", "antibody",
           "inverse agonist", "antagonist", "antisense oligonucleotide"}
ACTIVATE = {"agonist", "activator", "positive modulator", "potentiator"}

drug_targets = defaultdict(set)   # (drug, cls) -> set(gene_idx)
with open("dgidb_interactions.tsv", newline="", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    h = {n: i for i, n in enumerate(next(rd))}
    for row in rd:
        if len(row) < 13:
            continue
        g = idx.get(row[h["gene_name"]].strip().upper())
        if g is None:
            continue
        itype = (row[h["interaction_type"]] or "").strip().lower()
        if itype in INHIBIT:
            cls = -1
        elif itype in ACTIVATE:
            cls = 1
        else:
            continue
        drug = (row[h["drug_name"]] or row[h["drug_claim_name"]]).strip().upper()
        if drug:
            drug_targets[(drug, cls)].add(g)

A_pos = defaultdict(lambda: defaultdict(float))
A_neg = defaultdict(lambda: defaultdict(float))
for (drug, cls), targets in drug_targets.items():
    t = sorted(targets)
    k = len(t)
    if k < 2:
        continue
    w = 1.0 / np.sqrt(k)
    A = A_pos if cls > 0 else A_neg
    for a in range(k):
        ia = t[a]
        row_a = A[ia]
        for b in range(a + 1, k):
            ib = t[b]
            row_a[ib] += w
            A[ib][ia] += w

K = 32
ei, es, ew = [], [], []
for A, sign in ((A_pos, 1.0), (A_neg, -1.0)):
    for g, nbrs in A.items():
        top = sorted(nbrs.items(), key=lambda kv: -kv[1])[:K]
        for j, w in top:
            ei.append((g, j))
            es.append(sign)
            ew.append(w)

edge_index = np.array(ei, dtype=np.int32).T
edge_sign = np.array(es, dtype=np.float32)
edge_w = np.array(ew, dtype=np.float32)
np.savez("dgidb_graph.npz", edge_index=edge_index, edge_sign=edge_sign, edge_w=edge_w)
print("edges:", edge_index.shape[1], "genes involved:", len(set(edge_index.ravel().tolist())))
print("pos edges:", int((edge_sign > 0).sum()), "neg edges:", int((edge_sign < 0).sum()))
print("weight mean:", float(edge_w.mean()))
