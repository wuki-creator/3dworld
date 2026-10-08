#!/bin/bash
# Generate and prep tight-decode amp variants for the fid response probe.
set -x
export CUDA_VISIBLE_DEVICES=7
export TMPDIR=/nfs_beijing_os/zizhuo_vcc/work/tmp
CKPT=/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed113_np1.pt
EXTRACT=/home/zizhuo/vcc_data/extracted
PRED=/nfs_beijing_os/zizhuo_vcc/preds
LOG=/nfs_beijing_os/zizhuo_vcc/work/logs/gen_amp.log

for AMP in 4 16; do
  echo "=== predict amp=$AMP $(date) ===" >> $LOG
  python3 -u /nfs_beijing_os/zizhuo_vcc/work/v14_predict.py \
    --ckpt $CKPT --scale 0.5 --amp $AMP \
    --decode-style tight --jitter-shape 200 --cells-per-target 400 \
    --controls-dir $EXTRACT --genes $EXTRACT/gene_names.csv \
    --perts $EXTRACT/pert_counts.csv \
    --out $PRED/v13_tight_amp${AMP}.h5ad >> $LOG 2>&1
  echo "=== prep amp=$AMP $(date) ===" >> $LOG
  /home/zizhuo/.local/bin/vcc prep $PRED/v13_tight_amp${AMP}.h5ad \
    -g $EXTRACT/gene_names.csv --perts $EXTRACT/pert_counts.csv \
    -o $PRED/v13_tight_amp${AMP}.vcc >> $LOG 2>&1
  echo "=== amp=$AMP done $(date) ===" >> $LOG
done
