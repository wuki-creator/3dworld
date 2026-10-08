#!/bin/bash
# Wait for all vccsw-* runs, then proxy-eval every best.pt and log results.
OUT=/nfs_beijing_os/zizhuo_vcc/logs/seed_sweep_results.txt
: > "$OUT"
for i in $(seq 1 240); do
  N=$(systemctl --user list-units 'vccsw-*' --no-legend 2>/dev/null | grep -c running)
  echo "poll $i: $N still running" >> "$OUT"
  [ "$N" -eq 0 ] && break
  sleep 60
done
echo "=== evaluation ===" >> "$OUT"
cd /nfs_beijing_os/zizhuo_vcc/work
for s in 301 302 303 304 305 306 307 308 309 310 311 312; do
  CK=/nfs_beijing_os/zizhuo_vcc/ckpts/v13raw_s$s/best.pt
  if [ -f "$CK" ]; then
    R=$(CUDA_VISIBLE_DEVICES=3 python3 proxy_score.py --ckpt "$CK" 2>/dev/null | grep -E 'raw val cosine|scale=0.50' | tr '\n' ' ')
    echo "seed=$s $R" >> "$OUT"
  else
    echo "seed=$s MISSING" >> "$OUT"
  fi
done
echo "=== done $(date) ===" >> "$OUT"
