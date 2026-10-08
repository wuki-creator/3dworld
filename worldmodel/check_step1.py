# -*- coding: utf-8 -*-
"""检查 CollecTRI 下载与 01 核心管线进程状态"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "ls -la /nfs_beijing/zizhuo/vcc/data/resources/ 2>/dev/null; "
    "head -2 /nfs_beijing/zizhuo/vcc/data/resources/CollecTRI_regulons.csv 2>/dev/null; "
    "echo '--- core2.log ---'; "
    "tail -20 /nfs_beijing/zizhuo/vcc/results/core2.log 2>/dev/null || echo NO_LOG; "
    "echo '--- procs ---'; "
    "ps aux | grep -E '01_core|curl' | grep -v grep; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=90, verbose=True)
print("EXIT:", rc)
ch.close(); cli.close()
