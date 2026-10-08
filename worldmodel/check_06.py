# -*- coding: utf-8 -*-
"""检查 06 图版生成进度"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "tail -20 /nfs_beijing/zizhuo/vcc/results/figures2.log; "
    "ls -la /nfs_beijing/zizhuo/vcc/results/figures/ 2>/dev/null; "
    "ps aux | grep 06_figures | grep -v grep | wc -l; echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
