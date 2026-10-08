# -*- coding: utf-8 -*-
"""轮询 probe 日志（短命令，安全）"""
import sys
import time
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
for i in range(10):
    rc, out = run_remote(
        ch,
        "cat /nfs_beijing/zizhuo/vcc/results/probe_perts.log; echo __RC__$?",
        timeout=45)
    if "PROBE_DONE" in out:
        print(out)
        break
    time.sleep(20)
else:
    print(out)
ch.close(); cli.close()
