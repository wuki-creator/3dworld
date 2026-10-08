"""Build multi-prior per-gene feature matrix aligned to the VCC gene panel.

Sources:
  A. STRING physical links  -> PPI degree (score>=700), weighted degree, top10 mean score
  B. STRING full links      -> functional association degree, weighted degree, top10 mean
  C. Reactome pathways GMT  -> pathway count + 8-dim hashed pathway embedding
Output: prior_string_reactome.npy  (G x 17), float32
"""
import csv, gzip, zipfile, io
import numpy as np

panel = [r[0] for r in list(csv.reader(open("gene_names.csv", encoding="utf-8")))[1:] if r]
idx = {g: i for i, g in enumerate(panel)}
P = len(panel)

# ---- STRING id -> symbol ----
id2sym = {}
with gzip.open("priors/string_info.tsv.gz", "rt", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    next(rd)
    for row in rd:
        if len(row) >= 2 and row[1]:
            id2sym[row[0]] = row[1].upper()

def string_features(path_gz, score_thresh=700):
    deg = np.zeros(P); wdeg = np.zeros(P)
    tops = [[] for _ in range(P)]
    n_in = 0
    with gzip.open(path_gz, "rt", encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter=" ")
        next(rd)
        for row in rd:
            if len(row) < 3:
                continue
            g1 = id2sym.get(row[0]); g2 = id2sym.get(row[1])
            if g1 is None or g2 is None:
                continue
            i1 = idx.get(g1); i2 = idx.get(g2)
            if i1 is None or i2 is None:
                continue
            s = int(row[2]) / 1000.0
            n_in += 1
            if int(row[2]) >= score_thresh:
                deg[i1] += 1; deg[i2] += 1
            wdeg[i1] += s; wdeg[i2] += s
            if len(tops[i1]) < 12: tops[i1].append(s)
            if len(tops[i2]) < 12: tops[i2].append(s)
    top_mean = np.array([np.mean(sorted(t, reverse=True)[:10]) if t else 0.0 for t in tops])
    print(path_gz, "panel pairs:", n_in)
    return deg, wdeg, top_mean

d1, w1, t1 = string_features("priors/string_phys.gz")
d2, w2, t2 = string_features("priors/string_full.gz")

# ---- Reactome ----
pw_count = np.zeros(P)
rng = np.random.default_rng(7)
pw_emb = np.zeros((P, 8))
with zipfile.ZipFile("priors/reactome_gmt.zip") as z:
    name = [n for n in z.namelist() if n.endswith(".gmt")][0]
    with z.open(name) as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pw_id = parts[1] if parts[1] else parts[0]
            genes = [g.upper() for g in parts[2:] if g]
            vec = rng.standard_normal(8)  # deterministic per line order? NO - need per pathway id
            # use hash of pathway id for reproducibility
            import zlib; h = zlib.crc32(pw_id.encode("utf-8"))
            rng2 = np.random.default_rng(h)
            vec = rng2.standard_normal(8)
            hits = 0
            for g in genes:
                i = idx.get(g)
                if i is not None:
                    pw_count[i] += 1
                    pw_emb[i] += vec
                    hits += 1
            _ = hits
norm = np.linalg.norm(pw_emb, axis=1, keepdims=True).clip(min=1e-6)
pw_emb = pw_emb / norm
print("reactome genes covered:", int((pw_count > 0).sum()))

F = np.stack([
    np.log1p(d1), np.log1p(w1), t1,
    np.log1p(d2), np.log1p(w2), t2,
    np.log1p(pw_count),
], axis=1)
F = np.concatenate([F, pw_emb], axis=1).astype(np.float32)
np.save("prior_string_reactome.npy", F)
print("shape:", F.shape)
print("covered genes (any):", int((F[:, :6].sum(1) > 0).sum()))
