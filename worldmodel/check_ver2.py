# -*- coding: utf-8 -*-
"""精简核对：core2.log 时间/内容 + 服务器 01 是否含 xiehon"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "stat -c '%y' /nfs_beijing/zizhuo/vcc/results/core2.log; "
    "grep -c '=====' /nfs_beijing/zizhuo/vcc/results/core2.log; "
    "grep '=====' /nfs_beijing/zizhuo/vcc/results/core2.log; "
    "grep -c 'xiehon' /nfs_beijing/zizhuo/vcc/scripts/01_core_pipeline.py; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=90, verbose=True)
print(out)
print("EXIT:", rc)
ch.close(); cli.close()
