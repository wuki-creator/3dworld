"""Resumable PubChem fetch for approved-drug physchem (panel genes only).
Saves drug_physchem.json incrementally after each batch; rerun to resume.
"""
import json, os, time, urllib.request

PROPS = "MolecularWeight,XLogP,TPSA,RotatableBondCount"
URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/property/"
       + PROPS + "/JSON")

drugs = [d for d in open("panel_drugs_approved.txt", encoding="utf-8").read().split("\n") if d]
store = {}
if os.path.exists("drug_physchem.json"):
    store = json.load(open("drug_physchem.json"))
print("total %d, already have %d" % (len(drugs), len(store)))

def fetch_batch(names):
    body = json.dumps({"name": names}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0 vcc-prior"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    out = {}
    for rec in data.get("PropertyTable", {}).get("Properties", []):
        title = rec.get("Title")
        key = str(title).upper() if title else None
        if key:
            out[key] = [rec.get(p) for p in ("MolecularWeight", "XLogP", "TPSA", "RotatableBondCount")]
    return out

todo = [d for d in drugs if d not in store]
CH = 80
done = 0
for s in range(0, len(todo), CH):
    chunk = todo[s:s+CH]
    try:
        for k, v in fetch_batch(chunk).items():
            store[k] = v
    except Exception:
        for d in chunk:
            try:
                for k, v in fetch_batch([d]).items():
                    store[k] = v
            except Exception:
                pass
    done += len(chunk)
    if done % 400 < CH:
        json.dump(store, open("drug_physchem.json", "w"))
        print("progress %d/%d store=%d" % (done, len(todo), len(store)), flush=True)
    time.sleep(0.35)
json.dump(store, open("drug_physchem.json", "w"))
print("DONE store=%d" % len(store))
