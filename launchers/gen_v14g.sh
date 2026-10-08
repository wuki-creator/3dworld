#!/bin/bash
# Generate and prep v14g RAW-delta tight-decode amp1.0 (clip 3.0).
set -x
export CUDA_VISIBLE_DEVICES=7
export TMPDIR=/nfs_beijing_os/zizhuo_vcc/work/tmp
CKPT=/nfs_beijing_os/zizhuo_vcc/ckpts/v14g_full/best.pt
EXTRACT=/home/zizhuo/vcc_data/extracted
PRED=/nfs_beijing_os/zizhuo_vcc/preds
LOG=/nfs_beijing_os/zizhuo_vcc/work/logs/gen_v14g.log

echo "=== predict v14g raw amp=1.0 $(date) ===" >> $LOG
python3 -u /nfs_beijing_os/zizhuo_vcc/work/v14_predict.py \
  --ckpt $CKPT --top-k -1 --scale 1.0 --self-scale 1.0 --amp 1.0 --clip 3.0 \
  --decode-style tight --jitter-shape 200 --cells-per-target 400 \
  --controls-dir $EXTRACT --genes $EXTRACT/gene_names.csv \
  --perts $EXTRACT/pert_counts.csv \
  --out $PRED/v14g_raw_amp1.0_c3.h5ad >> $LOG 2>&1
echo "=== prep $(date) ===" >> $LOG
/home/zizhuo/.local/bin/vcc prep $PRED/v14g_raw_amp1.0_c3.h5ad \
  -g $EXTRACT/gene_names.csv --perts $EXTRACT/pert_counts.csv \
  -o $PRED/v14g_raw_amp1.0_c3.vcc >> $LOG 2>&1
echo "=== v14g done $(date) ===" >> $LOG
