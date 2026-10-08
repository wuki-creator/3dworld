# -*- coding: utf-8 -*-
"""只拉回缺失的 Fig7/Fig8"""
import os
import sys
import base64
import time
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

LOCAL_DIR = r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc\figures_check"
CH = 50000

cli, ch = open_channel(verbose=False)
for nm in ["Fig7_qc.jpg", "Fig8_ko.jpg"]:
    run_remote(ch, f"base64 -w 0 /tmp/figdl/{nm} > /tmp/figdl/{nm}.b64",
               timeout=60)
    rc, out = run_remote(ch, f"stat -c '%s' /tmp/figdl/{nm}.b64; echo __RC__$?",
                         timeout=30)
    b64_len = int([l for l in out.splitlines() if l.strip().isdigit()][0])
    chunks = []
    for off in range(0, b64_len, CH):
        rc, out = run_remote(
            ch,
            f"tail -c +{off+1} /tmp/figdl/{nm}.b64 | head -c {CH}; "
            f"echo; echo __RC__$?",
            timeout=60)
        lines = out.splitlines()
        body = [x.strip() for x in lines[1:]
                if x.strip() and not x.strip().startswith("__RC__")
                and not x.strip().startswith("zizhuo@")
                and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                        "0123456789+/=" for c in x.strip())]
        chunks.append("".join(body))
    data = base64.b64decode("".join(chunks))
    with open(os.path.join(LOCAL_DIR, nm), "wb") as f:
        f.write(data)
    print(nm, len(data), "bytes")
ch.close(); cli.close()
print("DONE")
