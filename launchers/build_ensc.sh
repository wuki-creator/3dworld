#!/bin/bash
# ensC = ensB 10 成员 + v25_noaux_s513 + v25_noaux_s514
cd /nfs_beijing_os/zizhuo_vcc/work
K=/nfs_beijing_os/zizhuo_vcc/ckpts
OLD=/home/zizhuo/cross_cell_vcc_v2/data/vcc_2026
P=/nfs_beijing_os/zizhuo_vcc/preds
export CUDA_VISIBLE_DEVICES=6
export TMPDIR=/nfs_beijing_os/zizhuo_vcc/work/tmp
python3 -u /nfs_beijing_os/zizhuo_vcc/work/ensemble_predict_h1.py \
  --ckpts $K/v14g_full/best.pt $K/v14g_s421/best.pt $K/v14g_s422/best.pt $K/v14g_s423/best.pt $K/v13_compat/magworld_h1_v13_full_seed113_np1.pt $K/v15_full/best.pt $K/v14d_graphk100/best.pt $K/v18dgidb_s441/best.pt $K/v18dgidb_s442/best.pt $K/v25_noaux_s512/best.pt $K/v25_noaux_s513/best.pt $K/v25_noaux_s514/best.pt \
  --top-k 500 --scale 0.5 --self-scale 1.0 --amp 1.0 --clip 8.0 \
  --decode-style tight --jitter-shape 200 --cells-per-target 400 \
  --controls-dir /nfs_beijing_os/zizhuo_vcc/valctx --genes $OLD/gene_names.csv --perts $OLD/pert_counts.csv \
  --val-signatures /nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz \
  --out $P/ensC_proper_k500s05c8.h5ad
echo PREDICT_EXIT=$?
