# -*- coding: utf-8 -*-
"""Register WorldModelH1V34 / WorldModelH1V35 in proxy_score.py and v14_predict.py (idempotent)."""
import ast

TARGETS = (
    "/nfs_beijing_os/zizhuo_vcc/work/proxy_score.py",
    "/nfs_beijing_os/zizhuo_vcc/work/v14_predict.py",
)
NEW = [
    ('("model_world_h1_v34", "WorldModelH1V34"),', "model_world_h1_v34"),
    ('("model_world_h1_v35", "WorldModelH1V35"),', "model_world_h1_v35"),
    ('("model_world_h1_v36", "WorldModelH1V36"),', "model_world_h1_v36"),
    ('("model_world_h1_v37", "WorldModelH1V37"),', "model_world_h1_v37"),
]
ANCHOR = '("model_world_h1_v33"'

for path in TARGETS:
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines()
    changed = False
    for line, tag in NEW:
        if tag in src:
            continue
        hit = None
        for i, ln in enumerate(lines):
            if ANCHOR in ln:
                hit = i
                break
        if hit is None:
            print(path, "ANCHOR NOT FOUND for", tag)
            continue
        indent = lines[hit][: len(lines[hit]) - len(lines[hit].lstrip())]
        lines.insert(hit + 1, indent + line)
        src = "\n".join(lines) + "\n"
        changed = True
        print(path, "registered", tag)
    if changed:
        ast.parse(src)
        open(path, "w", encoding="utf-8").write(src)
        ast.parse(open(path, encoding="utf-8").read())
        print(path, "syntax OK")
