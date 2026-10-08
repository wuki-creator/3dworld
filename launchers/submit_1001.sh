#!/bin/bash
# 10-01 submission plan:
#   slot 1: v13_raw_amp1.0_tight_c3  (amplitude probe between 0.5-best and 64-bad)
#   slot 2: v14g_raw_amp1.0_tight_c3  (v14g model candidate, same recipe)
export VCC_TOKEN=$VCC_TOKEN
export PATH=/home/zizhuo/.local/bin:$PATH
LOG=/nfs_beijing_os/zizhuo_vcc/logs/auto_submit_1001.log
PRED=/nfs_beijing_os/zizhuo_vcc/preds

submit_and_poll () {
  local FILE=$1 NAME=$2
  echo "=== submit $NAME $(date) ===" >> $LOG
  local OUT E ST i
  OUT=$(vcc submit "$FILE" -m "$NAME" 2>&1)
  echo "$OUT" >> $LOG
  E=$(echo "$OUT" | grep -oE 'entry: [A-Za-z0-9]+' | awk '{print $2}')
  if [ -n "$E" ]; then
    for i in $(seq 1 240); do
      ST=$(vcc status "$E" 2>&1 | grep -E '^  status:' | awk '{print $2}')
      echo "poll $i: $E status=$ST $(date +%H:%M:%S)" >> $LOG
      if [ "$ST" = "published" ] || [ "$ST" = "failed" ]; then
        vcc status "$E" >> $LOG 2>&1
        break
      fi
      sleep 60
    done
  fi
}

submit_and_poll $PRED/v13_raw_amp1.0_c3.vcc v13_raw_amp1.0_tight_c3
submit_and_poll $PRED/v14g_raw_amp1.0_c3.vcc v14g_raw_amp1.0_tight_c3
echo "=== done $(date) ===" >> $LOG
