#!/bin/bash
LOG=/nfs_beijing_os/zizhuo_vcc/logs/auto_submit_ensb.log
export VCC_TOKEN=$VCC_TOKEN
VCC=/home/zizhuo/.local/bin/vcc
E=/nfs_beijing_os/zizhuo_vcc/preds/ensB_raw_amp1.0_c3.vcc
CUR=3Ae3AYJ8ILstqUbJaurp

echo "=== wait for current entry $CUR $(date) ===" >> $LOG
for i in $(seq 1 240); do
  OUT=$($VCC status "$CUR" 2>&1)
  echo "$OUT" | grep -qiE 'complete|scored|failed|error' && { echo "current done: $OUT" >> $LOG; break; }
  echo "wait $i $(date +%H:%M:%S): $(echo "$OUT" | head -1 | cut -c1-80)" >> $LOG
  sleep 60
done

echo "=== submit ensB $(date) ===" >> $LOG
OUT=$($VCC submit "$E" -m ensB_raw_amp1.0_c3 2>&1)
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
