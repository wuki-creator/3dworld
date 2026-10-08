#!/bin/bash
# Proxy-eval v34-v37 as their checkpoints appear; results appended to proxy_v34_37.txt
R=/nfs_beijing_os/zizhuo_vcc/work/proxy_v34_37.txt
: > $R
cd /nfs_beijing_os/zizhuo_vcc/work
for name in v34 v35 v36 v37; do
  CK=/nfs_beijing_os/zizhuo_vcc/ckpts/${name}_s543/best.pt
  for i in $(seq 1 120); do [ -f "$CK" ] && break; sleep 20; done
  echo "=== PROXY $name $(date +%H:%M:%S) ===" >> $R
  CUDA_VISIBLE_DEVICES=6 timeout 280 python3 proxy_score.py --ckpt "$CK" 2>/dev/null \
    | grep -aE 'raw val cosine|scale=0.50|fid|reach|jaccard' >> $R
  echo "--- rc=$? $name" >> $R
done
echo ALL_PROXY_DONE >> $R
