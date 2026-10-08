# -*- coding: utf-8 -*-
"""杀掉旧 05 进程，启动 05 v4（magworld5.log）"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "pkill -f 05_magworld_model.py; sleep 1; "
    "setsid nohup /nfs_beijing/zizhuo/vcc/venv311/bin/python "
    "/nfs_beijing/zizhuo/vcc/scripts/05_magworld_model.py "
    "> /nfs_beijing/zizhuo/vcc/results/magworld5.log 2>&1 < /dev/null & disown; "
    "sleep 3; ps aux | grep 05_magworld | grep -v grep; echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
