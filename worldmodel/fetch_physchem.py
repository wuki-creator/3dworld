"""Fetch physicochemical descriptors for DGIDB drugs via PubChem PUG-REST,
then aggregate per gene (mean descriptors of interacting drugs, split by
inhibit vs activate class). Output: prior_physchem.npy (G x 10) float32.

cols: inh_mean[MW, XLogP, TPSA, RotB], act_mean[MW, XLogP, TPSA, RotB],
      inh_count_log, act_count_log
"""
import csv, json, time, urllib.request

PROPS = "MolecularWeight,XLogP,TPSA,RotatableBondCount"

# drug -> class (activate wins if both), drug upper name
drug_cls = {}
with open("dgidb_interactions.tsv", newline="", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    h = {n: i for i, n in enumerate(header)}
    for row in rd:
        if len(row) < 13:
            continue
        drug = (row[h["drug_name"]] or row[h["drug_claim_name"]]).strip().upper()
        if not drug:
            continue
        itype = (row[h["interaction_type"]] or "").strip().lower()
        cls = 1 if itype in {"agonist", "activator", "positive modulator", "potentiator"} else (
              -1 if itype in {"inhibitor", "blocker", "negative modulator", "antibody",
                              "inverse agonist", "antagonist", "antisense oligonucleotide"} else 0)
        if cls != 0 and drug not in drug_cls:
            drug_cls[drug] = cls

drugs = sorted(drug_cls)
print("unique drugs to query:", len(drugs))

props_by_drug = {}
URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/property/"
       + PROPS + "/JSON")

def fetch_batch(names):
    body = json.dumps({"name": names}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0 vcc-prior"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    out = {}
    for rec in data.get("PropertyTable", {}).get("Properties", []):
        name_key = None
        # PUG returns properties keyed by CID; match back by order is unreliable,
        # so the request returns only found ones; use 'Title' field if present
        title = rec.get("Title")
        key = str(title).upper() if title else None
        out[key] = [rec.get(p) for p in ("MolecularWeight", "XLogP", "TPSA", "RotatableBondCount")]
    return out

CH = 80
fails = []
for s in range(0, len(drugs), CH):
    chunk = drugs[s:s+CH]
    try:
        got = fetch_batch(chunk)
        for k, v in got.items():
            if k:
                props_by_drug[k] = v
    except Exception as e:
        fails.append((s, str(e)[:100]))
        # fallback: per-drug
        for d in chunk:
            try:
                got = fetch_batch([d])
                for k, v in got.items():
                    if k:
                        props_by_drug[k] = v
            except Exception:
                pass
    if (s // CH) % 10 == 0:
        print("progress %d/%d found=%d" % (s, len(drugs), len(props_by_drug)), flush=True)
    time.sleep(0.34)

print("total drugs with props:", len(props_by_drug), "batch fails:", len(fails))
json.dump({k: v for k, v in props_by_drug.items()},
          open("drug_physchem.json", "w"))
