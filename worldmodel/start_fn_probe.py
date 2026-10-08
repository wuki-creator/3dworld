# -*- coding: utf-8 -*-
"""后台启动 03 功能分析 v2 + probe_perts，立即返回"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "cd /nfs_beijing/zizhuo/vcc && "
    "setsid nohup venv311/bin/python scripts/03_pathway_tf_gsea.py "
    "> results/function9.log 2>&1 < /dev/null & disown; "
    "setsid nohup venv311/bin/python scripts/probe_perts.py "
    "> results/probe_perts.log 2>&1 < /dev/null & disown; "
    "sleep 3; ps aux | grep -E '03_pathway|probe_perts' | grep -v grep; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=60, verbose=True)
print(out)
print("EXIT:", rc)
ch.close(); cli.close()
