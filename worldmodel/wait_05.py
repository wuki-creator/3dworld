# -*- coding: utf-8 -*-
"""轮询 magworld.log 直到 MAGWORLD_DONE（或进程消失）"""
import sys
import time
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
for i in range(40):
    rc, out = run_remote(
        ch,
        "tail -12 /nfs_beijing/zizhuo/vcc/results/magworld.log; "
        "ps aux | grep 05_magworld | grep -v grep | wc -l; echo __RC__$?",
        timeout=45)
    lines = [l for l in out.splitlines() if l.strip()]
    print(f"--- poll {i} ---")
    print("\n".join(lines[-14:]))
    if "MAGWORLD_DONE" in out:
        print("MAGWORLD FINISHED")
        break
    if out.rstrip().splitlines()[-2:-1] == ["0"] and "05_magworld" not in out:
        # process count 0 and not done -> died
        if "0" in lines[-2]:
            print("PROCESS GONE")
            break
    time.sleep(90)
ch.close(); cli.close()
