# -*- coding: utf-8 -*-
"""把本地 scripts_server/ 下的脚本通过 SSH 通道以 base64 分块传到服务器。"""
import sys
import base64
import time
from ssh_server import open_channel, run_remote

LOCAL_DIR = r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc\scripts_server"
REMOTE_DIR = "/nfs_beijing/zizhuo/vcc/scripts"
CHUNK = 3500


def upload(cli, chan, local_path, remote_path):
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    run_remote(chan, f"rm -f {remote_path}.b64", timeout=30)
    for i in range(0, len(b64), CHUNK):
        chunk = b64[i:i + CHUNK]
        run_remote(chan, f"echo -n '{chunk}' >> {remote_path}.b64", timeout=30)
    rc, _ = run_remote(chan, f"base64 -d {remote_path}.b64 > {remote_path} && "
                            f"wc -c {remote_path}", timeout=60)
    print("uploaded", remote_path, "rc=", rc)


if __name__ == "__main__":
    cli, chan = open_channel(verbose=False)
    run_remote(chan, f"mkdir -p {REMOTE_DIR}", timeout=30)
    files = sys.argv[1:] or ["00_download_data.py", "01_core_pipeline.py",
                             "02_velocity.py", "03_pathway_tf_gsea.py",
                             "04_ccc.py", "05_magworld_model.py",
                             "06_figures.py"]
    for fn in files:
        upload(cli, chan, f"{LOCAL_DIR}\\{fn}", f"{REMOTE_DIR}/{fn}")
    chan.close()
    cli.close()
    print("ALL_UPLOADED")
