# -*- coding: utf-8 -*-
"""运行 06 图版组装（后台，日志 figures2.log）"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "setsid nohup /nfs_beijing/zizhuo/vcc/venv311/bin/python "
    "/nfs_beijing/zizhuo/vcc/scripts/06_figures.py "
    "> /nfs_beijing/zizhuo/vcc/results/figures2.log 2>&1 < /dev/null & disown; "
    "sleep 3; ps aux | grep 06_figures | grep -v grep; echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
