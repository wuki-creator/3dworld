#!/bin/bash
# Generate and prep RAW-delta tight-decode amp 1.0 and 4.0 probes (clip 3.0).
set -x
export CUDA_VISIBLE_DEVICES=7
export TMPDIR=/nfs_beijing_os/zizhuo_vcc/work/tmp
CKPT=/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed113_np1.pt
EXTRACT=/home/zizhuo/vcc_data/extracted
PRED=/nfs_beijing_os/zizhuo_vcc/preds
LOG=/nfs_beijing_os/zizhuo_vcc/work/logs/gen_raw_amp_small.log

for AMP in 1.0 4.0; do
  echo "=== predict raw amp=$AMP $(date) ===" >> $LOG
  python3 -u /nfs_beijing_os/zizhuo_vcc/work/v14_predict.py \
    --ckpt $CKPT --top-k -1 --scale 1.0 --self-scale 1.0 --amp $AMP --clip 3.0 \
    --decode-style tight --jitter-shape 200 --cells-per-target 400 \
    --controls-dir $EXTRACT --genes $EXTRACT/gene_names.csv \
    --perts $EXTRACT/pert_counts.csv \
    --out $PRED/v13_raw_amp${AMP}_c3.h5ad >> $LOG 2>&1
  echo "=== prep raw amp=$AMP $(date) ===" >> $LOG
  /home/zizhuo/.local/bin/vcc prep $PRED/v13_raw_amp${AMP}_c3.h5ad \
    -g $EXTRACT/gene_names.csv --perts $EXTRACT/pert_counts.csv \
    -o $PRED/v13_raw_amp${AMP}_c3.vcc >> $LOG 2>&1
  echo "=== raw amp=$AMP done $(date) ===" >> $LOG
done
