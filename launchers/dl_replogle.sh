#!/bin/bash
cd /nfs_beijing_os/zizhuo_vcc/pertseq/replogle
rm -f _std.txt.gz raw_counts_head_test.gz
for d in reploglek562genomewide reploglek562essential reploglerpe1essential; do
  for t in 1 2 3 4 5 6 7 8; do
    curl -sSL -C - --connect-timeout 15 -A "Mozilla/5.0" -o "${d}_std.txt.gz" \
      "https://maayanlab.cloud/static/hdfs/harmonizome/data/${d}/gene_attribute_matrix_standardized.txt.gz" && break
    sleep 4
  done
done
echo DONE_REPLOGLE
ls -la
