#!/bin/bash
export VCC_TOKEN=$VCC_TOKEN
export PATH=/home/zizhuo/.local/bin:$PATH
LOG=/nfs_beijing_os/zizhuo_vcc/logs/auto_submit_$(date +%m%d).log
echo "=== submit 1: tight $(date) ===" >> $LOG
vcc submit /nfs_beijing_os/zizhuo_vcc/preds/v13_tight.vcc -m "v13_tight_decode_j200" >> $LOG 2>&1
for i in $(seq 1 40); do
  sleep 60
  if grep -q "scoring started" $LOG; then break; fi
done
sleep 120
echo "=== submit 2: pure $(date) ===" >> $LOG
vcc submit /nfs_beijing_os/zizhuo_vcc/preds/v13_pure.vcc -m "v13_pure_poisson_decode" >> $LOG 2>&1
echo "=== done $(date) ===" >> $LOG
