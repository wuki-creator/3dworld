#!/bin/bash
# v14g recipe retrain with raw-cosine checkpoint selection.
# usage: v14g_retrain.sh <seed> <gpu> <outdir>
SEED=${1:-421}
GPU=${2:-3}
OUT=${3:-/nfs_beijing_os/zizhuo_vcc/ckpts/v14g_s421}
cd /nfs_beijing_os/zizhuo_vcc/work
export PYTHONPATH=/nfs_beijing_os/zizhuo_vcc/work
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT"
exec python3 -u train_v14_rawsel.py \
  --signatures /nfs_beijing_os/zizhuo_vcc/signatures/h1_trainval_signatures.npz \
  --official-perts /nfs_beijing_os/zizhuo_vcc/h1_2025/pert_counts_Training.csv \
  --gene-embeddings /nfs_beijing_os/zizhuo_vcc/embeddings/vcc_gene_embeddings_hybrid_256.npy \
  --magworld-src /home/zizhuo/maglab_deploy/src \
  --model-module model_world_h1_v14 --model-class WorldModelH1V14 \
  --out "$OUT/best.pt" \
  --holdout-mode none --epochs 60 --patience 999 \
  --batch-size 8 --learning-rate 1e-4 --weight-decay 1e-4 --top-k 100 \
  --lambda-l1 1.0 --lambda-mmd 0.1 --lambda-bce 0.1 --lambda-kernel 0.05 \
  --lambda-shared-bias 0.05 --bce-temperature 0.05 \
  --lambda-de 1.0 --lambda-cosine 0.5 --lambda-rank 0.1 --lambda-pds 0.05 \
  --lambda-yield 0.15 --rank-temperature 0.05 --pds-tau 0.15 \
  --yield-threshold 0.05 --yield-temperature 0.02 \
  --latent-dim 64 --hidden-dim 192 --d-dir 32 --graph-k 50 --n-layers 3 \
  --response-rank 64 --max-delta 1.5 --context-strength 0.25 --langevin-init 4.0 \
  --kernel-powers 4 --use-dipole --use-langevin \
  --seed "$SEED" \
  --init-from /nfs_beijing_os/zizhuo_vcc/ckpts/v4_seed337 --freeze-base \
  --select-raw
