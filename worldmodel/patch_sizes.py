# -*- coding: utf-8 -*-
"""Patch /tmp/seg_downloader.py FILES list with exact byte sizes."""
EXACT = {
    "1857_Obese_Climp63_ER.zip": 8269364549,
    "1857_Obese_Climp63_ER-tubules.zip": 6460719821,
    "1857_Obese_Climp63_ER-sheets.zip": 3665644267,
    "1857_Obese_Climp63_Mitochondria.zip": 3213352821,
    "1857_Obese_Climp63_Lipid-droplet.zip": 2102255104,
    "1857_Obese_Climp63_Plasma-membrane.zip": 1270461744,
}

path = "/tmp/seg_downloader.py"
text = open(path).read()
lines = text.splitlines(keepends=True)
out = []
in_files = False
for line in lines:
    if line.startswith("FILES"):
        in_files = True
        out.append(line)
        continue
    if in_files:
        if line.strip().startswith("]"):
            in_files = False
            out.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("("):
            name = stripped.split('"')[1]
            out.append('    ("%s", %d),\n' % (name, EXACT[name]))
            continue
    out.append(line)
open(path, "w").write("".join(out))
print("PATCHED")
print(open(path).read()[open(path).read().index("FILES"):open(path).read().index("FILES") + 400])
