#!/bin/bash
LOG=/nfs_beijing_os/zizhuo_vcc/logs/auto_submit_1002.log
export VCC_TOKEN=$VCC_TOKEN
VCC=/home/zizhuo/.local/bin/vcc
E=/nfs_beijing_os/zizhuo_vcc/preds/ensB_proper_k500s05c8.vcc

echo "=== submit ensB_proper $(date) ===" >> $LOG
if [ ! -f "$E" ]; then
  echo "MISSING $E - skipped" >> $LOG
  exit 1
fi
OUT=$($VCC submit "$E" -m ensB_proper_k500s05c8 2>&1)
echo "$OUT" >> $LOG
ID=$(echo "$OUT" | grep -oE 'entry [A-Za-z0-9]+' | awk '{print $2}')
[ -z "$ID" ] && ID=$(echo "$OUT" | grep -oE '\b[A-Za-z0-9]{20,}\b' | head -1)
echo "parsed id: $ID" >> $LOG
[ -z "$ID" ] && exit 1
for i in $(seq 1 120); do
  OUT=$($VCC status "$ID" 2>&1)
  ST=$(echo "$OUT" | grep -oiE 'status[=: ][a-z]+' | head -1)
  echo "poll $i: $ID $ST $(date +%H:%M:%S)" >> $LOG
  echo "$OUT" | grep -qiE 'complete|scored|failed|error' && { echo "$OUT" >> $LOG; break; }
  sleep 60
done
echo "=== done $(date) ===" >> $LOG
