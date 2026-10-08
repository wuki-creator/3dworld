# -*- coding: utf-8 -*-
"""启动 07 in silico KO 实验（后台）并验证脚本在服务器上"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "wc -l /nfs_beijing/zizhuo/vcc/scripts/07_insilico_ko.py; "
    "setsid nohup /nfs_beijing/zizhuo/vcc/venv311/bin/python "
    "/nfs_beijing/zizhuo/vcc/scripts/07_insilico_ko.py "
    "> /nfs_beijing/zizhuo/vcc/results/insilico_ko.log 2>&1 < /dev/null & disown; "
    "sleep 3; ps aux | grep 07_insilico | grep -v grep; echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
