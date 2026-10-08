# -*- coding: utf-8 -*-
"""上传 probe_dat17 后台运行并轮询"""
import sys
import time
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote
from upload_scripts import upload

cli, ch = open_channel(verbose=False)
upload(cli, ch,
       r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc\scripts_server\probe_dat17.py",
       "/nfs_beijing/zizhuo/vcc/scripts/probe_dat17.py")
run_remote(ch,
           "setsid nohup /nfs_beijing/zizhuo/vcc/venv311/bin/python "
           "/nfs_beijing/zizhuo/vcc/scripts/probe_dat17.py "
           "> /nfs_beijing/zizhuo/vcc/results/probe3.log 2>&1 < /dev/null & disown",
           timeout=30)
for i in range(15):
    rc, out = run_remote(
        ch, "tail -30 /nfs_beijing/zizhuo/vcc/results/probe3.log; echo __RC__$?",
        timeout=45)
    if "PROBE3_DONE" in out:
        print(out)
        break
    time.sleep(20)
ch.close(); cli.close()
