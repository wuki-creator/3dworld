# -*- coding: utf-8 -*-
"""启动 01 核心管线（含 xiehon）后台运行并验证进程存活"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "cd /nfs_beijing/zizhuo/vcc && "
    "setsid nohup venv311/bin/python scripts/01_core_pipeline.py "
    "> results/core3.log 2>&1 < /dev/null & disown; "
    "sleep 5; "
    "ps aux | grep 01_core | grep -v grep; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=90, verbose=True)
print(out)
print("EXIT:", rc)
ch.close(); cli.close()
