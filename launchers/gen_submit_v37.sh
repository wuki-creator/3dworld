#!/bin/bash
# v37 final: proxy re-check -> predict -> prep -> submit -> poll
set -x
CKPT=/nfs_beijing_os/zizhuo_vcc/ckpts/v37_s543/best.pt
EXTRACT=/home/zizhuo/vcc_data/extracted
PRED=/nfs_beijing_os/zizhuo_vcc/preds
LOG=/nfs_beijing_os/zizhuo_vcc/logs/gen_v37.log
export CUDA_VISIBLE_DEVICES=7
export TMPDIR=/nfs_beijing_os/zizhuo_vcc/work/tmp
export VCC_TOKEN=$VCC_TOKEN
VCC=/home/zizhuo/.local/bin/vcc
cd /nfs_beijing_os/zizhuo_vcc/work

echo "=== proxy final v37 $(date) ===" > $LOG
timeout 280 python3 proxy_score.py --ckpt $CKPT 2>/dev/null | grep -aE 'raw val cosine|scale=0.50' > $PRED/proxy_v37_final.txt

echo "=== predict v37 $(date) ===" >> $LOG
python3 -u v14_predict.py \
  --ckpt $CKPT --top-k -1 --scale 1.0 --self-scale 1.0 --amp 1.0 --clip 3.0 \
  --decode-style tight --jitter-shape 200 --cells-per-target 400 \
  --controls-dir $EXTRACT --genes $EXTRACT/gene_names.csv \
  --perts $EXTRACT/pert_counts.csv \
  --out $PRED/v37_s543_raw_amp1.0_c3.h5ad >> $LOG 2>&1
echo "predict rc=$?" >> $LOG

echo "=== prep $(date) ===" >> $LOG
$VCC prep $PRED/v37_s543_raw_amp1.0_c3.h5ad \
  -g $EXTRACT/gene_names.csv --perts $EXTRACT/pert_counts.csv \
  -o $PRED/v37_s543_raw_amp1.0_c3.vcc >> $LOG 2>&1
echo "prep rc=$?" >> $LOG

echo "=== submit $(date) ===" >> $LOG
OUT=$($VCC submit $PRED/v37_s543_raw_amp1.0_c3.vcc -m v37_s543_raw_amp1.0_c3 2>&1)
echo "$OUT" >> $LOG
ID=$(echo "$OUT" | grep -oE 'entry [A-Za-z0-9]+' | awk '{print $2}')
echo "parsed id: $ID" >> $LOG
[ -z "$ID" ] && { echo NO_ENTRY_ABORT >> $LOG; exit 1; }

for i in $(seq 1 100); do
  OUT=$($VCC status $ID 2>&1)
  echo "poll $i $(date +%H:%M:%S)" >> $LOG
  echo "$OUT" | grep -aE 'Overall|Rank|status:|model:' >> $LOG
  echo "$OUT" | grep -qiE 'published|failed|error' && break
  sleep 60
done
echo GEN_SUBMIT_DONE >> $LOG
