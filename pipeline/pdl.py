# -*- coding: utf-8 -*-
"""服务器侧多线程分段下载器 v2 (断点续传 + 校验)。
用法: python3 pdl.py <url> <out> [threads]
连接被掐断时按已下载偏移续传; 校验每段最终长度。
"""
import os
import sys
import time
import threading
import urllib.request

url, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 16

req = urllib.request.Request(url, method="HEAD")
with urllib.request.urlopen(req, timeout=60) as r:
    final_url = r.geturl()
    size = int(r.headers["Content-Length"])
print(f"size={size} final={final_url[:120]}", flush=True)

part = (size + N - 1) // N
lock = threading.Lock()
status = [0] * N   # 0=ok -1=fail
t0 = time.time()


def fetch(i):
    lo = i * part
    hi = min(size, lo + part) - 1
    if lo > hi:
        status[i] = 0
        return
    path = f"{out}.part{i}"
    for attempt in range(30):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        want = hi - lo + 1
        if have >= want:
            status[i] = 0
            return
        try:
            req = urllib.request.Request(final_url)
            req.add_header("Range", f"bytes={lo + have}-{hi}")
            with urllib.request.urlopen(req, timeout=120) as r:
                code = getattr(r, "status", 200)
                if have and code != 206:
                    print(f"part{i}: no 206 (code={code}), wait", flush=True)
                    time.sleep(5)
                    continue
                mode = "ab" if have else "wb"
                with open(path, mode) as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
        except Exception as e:
            print(f"part{i} attempt{attempt}: {type(e).__name__}", flush=True)
            time.sleep(3)
    status[i] = -1


threads = [threading.Thread(target=fetch, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
last = 0.0
while any(t.is_alive() for t in threads):
    time.sleep(15)
    got = sum(os.path.getsize(f"{out}.part{i}") for i in range(N)
              if os.path.exists(f"{out}.part{i}"))
    el = time.time() - t0
    if got - last > 0:
        print(f"{got/1e6:.1f}/{size/1e6:.1f} MB  avg {got/1e6/el:.2f} MB/s",
              flush=True)
        last = got
for t in threads:
    t.join()
if any(s < 0 for s in status):
    print("FAILED_PARTS", status, flush=True)
    sys.exit(1)
total = 0
with open(out, "wb") as f:
    for i in range(N):
        p = f"{out}.part{i}"
        if os.path.exists(p):
            sz = os.path.getsize(p)
            with open(p, "rb") as g:
                while True:
                    buf = g.read(1 << 24)
                    if not buf:
                        break
                    f.write(buf)
            total += sz
            os.remove(p)
print(f"MERGED {total} bytes (expect {size})", flush=True)
if total != size:
    print("SIZE_MISMATCH", flush=True)
    sys.exit(1)
print(f"DONE {out} in {time.time()-t0:.0f}s", flush=True)
