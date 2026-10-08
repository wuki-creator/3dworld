# -*- coding: utf-8 -*-
"""等待 core3 完成（轮询 ALL_CORE_DONE），完成后跑 perturbation 探查"""
import sys
import time
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
for attempt in range(40):
    rc, out = run_remote(
        ch,
        "tail -4 /nfs_beijing/zizhuo/vcc/results/core3.log; echo __RC__$?",
        timeout=60)
    print(f"--- poll {attempt} ---")
    print(out[-500:])
    if "ALL_CORE_DONE" in out:
        print("CORE3 FINISHED")
        break
    time.sleep(60)
ch.close(); cli.close()
