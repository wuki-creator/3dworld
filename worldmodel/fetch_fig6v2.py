# -*- coding: utf-8 -*-
"""把服务器上的 Fig6_benchmark_v2.png 拉回本地 figures_check/"""
import sys, base64, os
sys.path.insert(0, '.')
from ssh_server import open_channel, run_remote

REMOTE = "/nfs_beijing/zizhuo/vcc/results/magworld/figures_v2/Fig6_benchmark_v2.png"
LOCAL = r"figures_check\Fig6_benchmark_v2.png"

cli, chan = open_channel()

# 先缩到宽 1600 再 base64
rc, text = run_remote(chan,
    "python3 -c \"from PIL import Image; import base64,io; "
    "im=Image.open('%s'); im.thumbnail((1600,1600)); b=io.BytesIO(); im.save(b,'PNG'); "
    "open('/tmp/f6.b64','w').write(base64.b64encode(b.getvalue()).decode())\" ; "
    "wc -c /tmp/f6.b64" % REMOTE, timeout=60)
print(text.splitlines()[-2:])

rc, text = run_remote(chan, "cat /tmp/f6.b64", timeout=60)
# 从输出中提取 base64 文本
lines = text.splitlines()
b64 = "".join(l.strip() for l in lines if l.strip() and not l.startswith(("cat ", "/nfs", "zizhuo@", "[")))
data = base64.b64decode(b64)
os.makedirs("figures_check", exist_ok=True)
with open(LOCAL, "wb") as f:
    f.write(data)
print("saved", LOCAL, len(data), "bytes")
cli.close()
