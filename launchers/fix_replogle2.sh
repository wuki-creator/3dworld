#!/bin/bash
cd /nfs_beijing_os/zizhuo_vcc/pertseq/replogle
declare -A TOTAL=( [reploglek562genomewide]=265909548 [reploglek562essential]=76812633 [reploglerpe1essential]=70151083 )
URLBASE=https://maayanlab.cloud/static/hdfs/harmonizome/data
NPAR=8
for round in $(seq 1 20); do
  all_done=1
  for d in reploglek562genomewide reploglek562essential reploglerpe1essential; do
    f=${d}_std.txt.gz
    T=${TOTAL[$d]}
    s=$(stat -c %s "$f" 2>/dev/null || echo 0)
    rem=$((T - s))
    if [ "$rem" -le 0 ]; then continue; fi
    all_done=0
    chunk=$(( (rem + NPAR - 1) / NPAR ))
    pids=()
    for i in $(seq 0 $((NPAR-1))); do
      cs=$((s + i*chunk))
      ce=$((cs + chunk - 1))
      [ $ce -gt $((T-1)) ] && ce=$((T-1))
      [ $cs -gt $ce ] && continue
      ( for t in 1 2 3 4 5; do
          curl -sSL --connect-timeout 15 -A "Mozilla/5.0" -r ${cs}-${ce} -o ${d}_rp${i} \
            "$URLBASE/${d}/gene_attribute_matrix_standardized.txt.gz" && break
          sleep 4
        done ) &
      pids+=($!)
    done
    wait "${pids[@]}"
    for i in $(seq 0 $((NPAR-1))); do
      if [ -f ${d}_rp${i} ]; then cat ${d}_rp${i} >> "$f"; rm -f ${d}_rp${i}; fi
    done
  done
  [ "$all_done" -eq 1 ] && break
done
for d in reploglek562genomewide reploglek562essential reploglerpe1essential; do
  f=${d}_std.txt.gz; T=${TOTAL[$d]}; s=$(stat -c %s "$f")
  ok=BROKEN; [ "$s" -eq "$T" ] && ok=OK
  gzip -t "$f" 2>/dev/null && gz=GZOK || gz=GZBAD
  echo "$d size=$s/$T $ok $gz"
done
echo DONE_FIXALL
