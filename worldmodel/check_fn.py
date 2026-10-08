# -*- coding: utf-8 -*-
"""检查 03 功能分析进程与产出状态"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "echo '=== 03 proc ==='; "
    "ps aux | grep 03_function | grep -v grep || echo NO_PROC; "
    "echo '=== function results ==='; "
    "ls -la /nfs_beijing/zizhuo/vcc/results/function/ 2>/dev/null; "
    "echo '=== full log grep ==='; "
    "grep -E 'DONE|SKIP|ERROR|Traceback|FAILED' /nfs_beijing/zizhuo/vcc/results/function8.log | tail -30; "
    "echo '=== zenodo collectri test ==='; "
    "timeout 30 curl -sIL 'https://zenodo.org/record/8192729/files/CollecTRI_regulons.csv?download=1' 2>&1 | head -8; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=120, verbose=True)
ch.close(); cli.close()
print("EXIT:", rc)
