# -*- coding: utf-8 -*-
"""1) zenodo 下载 CollecTRI 到本地资源目录并查看格式
2) 后台启动 01 核心管线（含 xiehon）"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "mkdir -p /nfs_beijing/zizhuo/vcc/data/resources && "
    "cd /nfs_beijing/zizhuo/vcc/data/resources && "
    "for i in 1 2 3 4 5; do "
    "curl -sL --max-time 300 -o CollecTRI_regulons.csv "
    "'https://zenodo.org/records/8192729/files/CollecTRI_regulons.csv?download=1' && break; "
    "echo retry $i; sleep 5; done; "
    "ls -la CollecTRI_regulons.csv; "
    "head -3 CollecTRI_regulons.csv; "
    "wc -l CollecTRI_regulons.csv; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=280, verbose=True)
print("EXIT:", rc)

# 后台启动 01 核心管线
cmd2 = (
    "cd /nfs_beijing/zizhuo/vcc && "
    "setsid nohup /nfs_beijing/zizhuo/vcc/venv311/bin/python scripts/01_core_pipeline.py "
    "> results/core2.log 2>&1 < /dev/null & disown; "
    "sleep 2; ps aux | grep 01_core | grep -v grep | head -3; echo __RC__$?"
)
rc2, out2 = run_remote(ch, cmd2, timeout=60, verbose=True)
print("EXIT2:", rc2)
ch.close(); cli.close()
