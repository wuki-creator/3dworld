# -*- coding: utf-8 -*-
"""查看 magworld3.log 实验一结果 + 重启修复后的 05"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "grep -E 'ridge|shared|delta_pearson|MAGWorld|MeanDelta|Identity|DONE' "
    "/nfs_beijing/zizhuo/vcc/results/magworld3.log | tail -20; echo __RC__$?",
    timeout=60)
print(out)
rc, out = run_remote(
    ch,
    "setsid nohup /nfs_beijing/zizhuo/vcc/venv311/bin/python "
    "/nfs_beijing/zizhuo/vcc/scripts/05_magworld_model.py "
    "> /nfs_beijing/zizhuo/vcc/results/magworld4.log 2>&1 < /dev/null & disown; "
    "sleep 2; echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
