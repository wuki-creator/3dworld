import os

import gears

pkg = os.path.dirname(gears.__file__)
for fname, keys in [
    ("gears.py", ["def predict", "def train", "gene2go", "def model_initialize",
                  "default_pert_graph", "uncertainty", "def GI_predict"]),
    ("utils.py", ["def dataverse_download", "def zip_data_download_wrapper"]),
]:
    src = open(os.path.join(pkg, fname), encoding="utf-8").read()
    lines = src.splitlines()
    hits = [i for i, l in enumerate(lines)
            if any(k in l for k in keys)]
    printed = set()
    for h in hits:
        for j in range(max(0, h - 2), min(len(lines), h + 40)):
            if j not in printed:
                print("%s L%d: %s" % (fname, j, lines[j]))
                printed.add(j)
        print("   ...")
