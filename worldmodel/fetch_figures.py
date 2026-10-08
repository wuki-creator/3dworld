# -*- coding: utf-8 -*-
"""从服务器拉回图版到本地目检。
流程：服务器端 PIL 压缩成 JPG -> base64 分块读回 -> 本地解码保存"""
import os
import sys
import base64
import time
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

LOCAL_DIR = r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc\figures_check"
os.makedirs(LOCAL_DIR, exist_ok=True)

cli, ch = open_channel(verbose=False)

# 1) 服务器端压缩
rc, out = run_remote(
    ch,
    "mkdir -p /tmp/figdl && /nfs_beijing/zizhuo/vcc/venv311/bin/python -c \""
    "import glob,os;"
    "from PIL import Image;"
    "files=sorted(glob.glob('/nfs_beijing/zizhuo/vcc/results/figures/Fig*.png'));"
    "print('NFILES',len(files));"
    "[(lambda im,p: (im.thumbnail((1800,1800)), im.convert('RGB').save(p,'JPEG',quality=82), print(p,os.path.getsize(p))))(Image.open(f),'/tmp/figdl/'+os.path.basename(f).replace('.png','.jpg')) for f in files]"
    "\"; ls -la /tmp/figdl/; echo __RC__$?",
    timeout=240)
print(out[-800:])

# 2) 逐文件 base64 读回
rc, out = run_remote(ch, "ls -1 --color=never /tmp/figdl/; echo __RC__$?", timeout=30)
names = [l.strip() for l in out.splitlines()
         if l.strip().endswith(".jpg") and "__RC__" not in l]
print("fetching:", names)

for nm in names:
    run_remote(ch, f"base64 -w 0 /tmp/figdl/{nm} > /tmp/figdl/{nm}.b64",
               timeout=60)
    # 文件大小
    rc, out = run_remote(ch, f"stat -c '%s' /tmp/figdl/{nm}.b64; echo __RC__$?",
                         timeout=30)
    size = int([l for l in out.splitlines() if l.strip().isdigit()][0])
    b64_len = size
    CH = 50000
    chunks = []
    t0 = time.time()
    for off in range(0, b64_len, CH):
        rc, out = run_remote(
            ch,
            f"tail -c +{off+1} /tmp/figdl/{nm}.b64 2>/dev/null | head -c {CH}; "
            f"echo; echo __RC__$?",
            timeout=30)
        lines = out.splitlines()
        # lines[0]=命令回显; 数据行为纯 base64; 其后可能有 __RC__ 标记和 shell 提示符
        body = [x.strip() for x in lines[1:]
                if x.strip() and not x.strip().startswith("__RC__")
                and not x.strip().startswith("zizhuo@")
                and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in x.strip())]
        chunks.append("".join(body))
    data = base64.b64decode("".join(chunks))
    local = os.path.join(LOCAL_DIR, nm)
    with open(local, "wb") as f:
        f.write(data)
    print(f"{nm}: {size}B b64->{len(data)}B in {time.time()-t0:.0f}s")

ch.close(); cli.close()
print("FETCH_DONE")
