# -*- coding: utf-8 -*-
"""运行探查脚本并读回日志"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "cd /nfs_beijing/zizhuo/vcc && "
    "venv311/bin/python scripts/probe_perts.py > results/probe_perts.log 2>&1; "
    "cat results/probe_perts.log; echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=240, verbose=True)
print(out)
print("EXIT:", rc)
ch.close(); cli.close()
