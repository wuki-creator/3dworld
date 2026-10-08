# -*- coding: utf-8 -*-
"""查看 KO 实验输出 CSV 的实际内容"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
rc, out = run_remote(
    ch,
    "head -5 /nfs_beijing/zizhuo/vcc/results/magworld/insilico_ko_tf_activity.csv; "
    "echo; cat /nfs_beijing/zizhuo/vcc/results/magworld/insilico_ko_summary.csv; "
    "echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
