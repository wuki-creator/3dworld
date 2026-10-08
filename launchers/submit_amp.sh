#!/bin/bash
# Submit raw_amp64 then raw_amp256 sequentially (wait for the first to finish scoring).
export VCC_TOKEN=$VCC_TOKEN
export PATH=/home/zizhuo/.local/bin:$PATH
LOG=/nfs_beijing_os/zizhuo_vcc/logs/auto_submit_0930.log
PRED=/nfs_beijing_os/zizhuo_vcc/preds

echo "=== submit raw_amp64 $(date) ===" >> $LOG
OUT1=$(vcc submit $PRED/v13_raw_amp64_c3.vcc -m "v13_raw_amp64_tight_c3" 2>&1)
echo "$OUT1" >> $LOG
E1=$(echo "$OUT1" | grep -oE 'entry: [A-Za-z0-9]+' | awk '{print $2}')
if [ -n "$E1" ]; then
  for i in $(seq 1 180); do
    ST=$(vcc status $E1 2>&1 | grep -E '^  status:' | awk '{print $2}')
    echo "poll $i: $E1 status=$ST $(date +%H:%M:%S)" >> $LOG
    if [ "$ST" = "published" ] || [ "$ST" = "failed" ]; then break; fi
    sleep 60
  done
fi
echo "=== submit raw_amp256 $(date) ===" >> $LOG
vcc submit $PRED/v13_raw_amp256_c3.vcc -m "v13_raw_amp256_tight_c3" >> $LOG 2>&1
echo "=== done $(date) ===" >> $LOG
