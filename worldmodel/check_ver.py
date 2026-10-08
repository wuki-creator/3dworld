# -*- coding: utf-8 -*-
"""确认 core2.log 时间戳 + 服务器上 01 脚本版本是否含 xiehon"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "stat -c '%y %n' /nfs_beijing/zizhuo/vcc/results/core2.log; "
    "grep -n 'xiehon' /nfs_beijing/zizhuo/vcc/scripts/01_core_pipeline.py || echo NO_XIEHON_IN_SCRIPT; "
    "head -5 /nfs_beijing/zizhuo/vcc/results/core2.log; "
    "ls -la /nfs_beijing/zizhuo/vcc/scripts/; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=90, verbose=True)
print("EXIT:", rc)
ch.close(); cli.close()
