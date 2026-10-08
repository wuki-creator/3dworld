# -*- coding: utf-8 -*-
"""读取实验一 LOO 基准结果"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "cat /nfs_beijing/zizhuo/vcc/results/magworld/loo_benchmark.csv; "
    "echo; cat /nfs_beijing/zizhuo/vcc/results/magworld/perturbation_discrimination.csv; "
    "echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
