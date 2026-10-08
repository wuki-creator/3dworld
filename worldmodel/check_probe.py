# -*- coding: utf-8 -*-
"""通用：检查 probe 进程/日志状态"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "ps aux | grep probe_perts | grep -v grep; "
    "ls -la /nfs_beijing/zizhuo/vcc/results/probe_perts.log 2>/dev/null; "
    "cat /nfs_beijing/zizhuo/vcc/results/probe_perts.log 2>/dev/null; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=60, verbose=True)
print(out)
print("EXIT:", rc)
ch.close(); cli.close()
