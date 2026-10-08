#!/bin/bash
# Wait for the tight submission to finish scoring, then submit pure.
export VCC_TOKEN=$VCC_TOKEN
export PATH=/home/zizhuo/.local/bin:$PATH
LOG=/nfs_beijing_os/zizhuo_vcc/logs/auto_submit_0929b.log
TIGHT=8uMsl6RU8qRQ3c0z7nMU
echo "=== waiting for $TIGHT $(date) ===" >> $LOG
for i in $(seq 1 180); do
  ST=$(vcc status $TIGHT 2>&1 | grep -E '^  status:' | awk '{print $2}')
  echo "poll $i: status=$ST $(date +%H:%M:%S)" >> $LOG
  if [ "$ST" = "published" ] || [ "$ST" = "scored" ] || [ "$ST" = "failed" ]; then
    break
  fi
  sleep 60
done
echo "=== submit pure $(date) ===" >> $LOG
vcc submit /nfs_beijing_os/zizhuo_vcc/preds/v13_pure.vcc -m "v13_pure_poisson_decode" >> $LOG 2>&1
echo "=== done $(date) ===" >> $LOG
