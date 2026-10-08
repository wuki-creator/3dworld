# -*- coding: utf-8 -*-
"""用绝对路径后台重启 probe_perts"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "setsid nohup /nfs_beijing/zizhuo/vcc/venv311/bin/python "
    "/nfs_beijing/zizhuo/vcc/scripts/probe_perts.py "
    "> /nfs_beijing/zizhuo/vcc/results/probe_perts.log 2>&1 < /dev/null & disown; "
    "sleep 3; ps aux | grep probe_perts | grep -v grep; echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=60, verbose=True)
print(out)
print("EXIT:", rc)
ch.close(); cli.close()
