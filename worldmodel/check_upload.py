# -*- coding: utf-8 -*-
"""核对上传的 03/06 脚本大小"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "wc -c /nfs_beijing/zizhuo/vcc/scripts/03_pathway_tf_gsea.py "
    "/nfs_beijing/zizhuo/vcc/scripts/06_figures.py; "
    "tail -3 /nfs_beijing/zizhuo/vcc/results/core3.log; "
    "ps aux | grep 01_core | grep -v grep | awk '{print $2, $10}'; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=90, verbose=True)
print(out)
print("EXIT:", rc)
ch.close(); cli.close()
