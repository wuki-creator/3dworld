# -*- coding: utf-8 -*-
"""查看 core2.log 中 xiehon 段落与 core 产物"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "grep -A6 'xiehon' /nfs_beijing/zizhuo/vcc/results/core2.log | head -30; "
    "echo '=== core files ==='; "
    "ls -la /nfs_beijing/zizhuo/vcc/results/core/; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=90, verbose=True)
print("EXIT:", rc)
ch.close(); cli.close()
