# -*- coding: utf-8 -*-
"""诊断分块读取：查看前两块实际返回内容"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
nm = "Fig1_overview.jpg"
run_remote(ch, f"base64 -w 0 /tmp/figdl/{nm} > /tmp/figdl/{nm}.b64", timeout=60)
rc, out = run_remote(ch, f"stat -c '%s' /tmp/figdl/{nm}.b64; echo __RC__$?",
                     timeout=30)
print("SIZE:", out)
for off in (0, 6000):
    rc, out = run_remote(
        ch,
        f"tail -c +{off+1} /tmp/figdl/{nm}.b64 | head -c 6000; echo __RC__$?",
        timeout=30)
    lines = out.splitlines()
    print(f"--- off={off} nlines={len(lines)} ---")
    for i, l in enumerate(lines[:4]):
        print(f"  line{i}: {repr(l[:80])}")
    print(f"  last: {repr(lines[-1][:60]) if lines else 'NONE'}")
ch.close(); cli.close()
