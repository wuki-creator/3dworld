# -*- coding: utf-8 -*-
"""单次检查 magworld 训练进度"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "tail -15 /nfs_beijing/zizhuo/vcc/results/magworld5.log; "
    "ps aux | grep 05_magworld | grep -v grep | wc -l; echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
