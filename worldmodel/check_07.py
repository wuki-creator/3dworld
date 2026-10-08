# -*- coding: utf-8 -*-
"""检查 07 KO 实验进度"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "tail -15 /nfs_beijing/zizhuo/vcc/results/insilico_ko.log; "
    "ps aux | grep 07_insilico | grep -v grep | wc -l; echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
