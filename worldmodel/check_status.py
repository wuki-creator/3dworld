# -*- coding: utf-8 -*-
"""一次性状态检查：function8 / velplots2 / xiehon 下载 / magworld 状态"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "echo '=== function8.log tail ==='; "
    "tail -25 /nfs_beijing/zizhuo/vcc/results/function8.log 2>/dev/null || echo MISSING; "
    "echo '=== velplots2.log tail ==='; "
    "tail -15 /nfs_beijing/zizhuo/vcc/results/velplots2.log 2>/dev/null || echo MISSING; "
    "echo '=== xiehon size ==='; "
    "stat -c '%s' /nfs_beijing/zizhuo/vcc/data/xiehon.h5ad 2>/dev/null || echo NOFILE; "
    "cat /nfs_beijing/zizhuo/vcc/results/dl_xiehon.status 2>/dev/null; "
    "echo '=== magworld dir ==='; "
    "ls -la /nfs_beijing/zizhuo/vcc/results/magworld/ 2>/dev/null; "
    "echo '=== running procs ==='; "
    "ps aux | grep -E 'python|wget|curl' | grep -v grep | head -20; "
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=120, verbose=True)
ch.close(); cli.close()
print("EXIT:", rc)
