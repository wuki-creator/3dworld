# -*- coding: utf-8 -*-
"""上传 05/06，检查 03 进度"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote
from upload_scripts import upload

cli, ch = open_channel(verbose=False)
upload(cli, ch,
       r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc\scripts_server\05_magworld_model.py",
       "/nfs_beijing/zizhuo/vcc/scripts/05_magworld_model.py")
upload(cli, ch,
       r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc\scripts_server\06_figures.py",
       "/nfs_beijing/zizhuo/vcc/scripts/06_figures.py")
rc, out = run_remote(
    ch,
    "grep -E 'DONE|skip' /nfs_beijing/zizhuo/vcc/results/function9.log | tail -8; "
    "ps aux | grep 03_pathway | grep -v grep | awk '{print $2}'; echo __RC__$?",
    timeout=60)
print(out)
ch.close(); cli.close()
