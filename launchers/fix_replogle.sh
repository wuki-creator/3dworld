#!/bin/bash
cd /nfs_beijing_os/zizhuo_vcc/pertseq/replogle
declare -A TOTAL=( [reploglek562genomewide]=265909548 [reploglek562essential]=76812633 [reploglerpe1essential]=70151083 )
for round in 1 2 3 4 5 6 7 8 9 10 11 12; do
  all_done=1
  for d in reploglek562genomewide reploglek562essential reploglerpe1essential; do
    f=${d}_std.txt.gz
    T=${TOTAL[$d]}
    s=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$s" -lt "$T" ]; then
      all_done=0
      e=$((T-1))
      curl -sSL --connect-timeout 15 -A "Mozilla/5.0" -r ${s}-${e} -o "${d}_tail.bin" \
        "https://maayanlab.cloud/static/hdfs/harmonizome/data/${d}/gene_attribute_matrix_standardized.txt.gz"
      got=$(stat -c %s "${d}_tail.bin" 2>/dev/null || echo 0)
      if [ "$got" -gt 0 ]; then
        cat "${d}_tail.bin" >> "$f"
        rm -f "${d}_tail.bin"
      fi
      sleep 2
    fi
  done
  [ "$all_done" -eq 1 ] && break
  sleep 3
done
for d in reploglek562genomewide reploglek562essential reploglerpe1essential; do
  f=${d}_std.txt.gz
  T=${TOTAL[$d]}
  s=$(stat -c %s "$f")
  ok=BROKEN; [ "$s" -eq "$T" ] && ok=OK
  gzip -t "$f" 2>/dev/null && gz=GZOK || gz=GZBAD
  echo "$d size=$s/$T $ok $gz"
done
echo DONE_FIXALL
