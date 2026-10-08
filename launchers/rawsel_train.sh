#!/bin/bash
# v13 retrain, old geometry, checkpoint selected by RAW val cosine.
# usage: rawsel_train.sh <seed> <gpu> <outdir>
SEED=${1:-113}
GPU=${2:-5}
OUT=${3:-/nfs_beijing_os/zizhuo_vcc/ckpts/v13raw_s113}
cd /nfs_beijing_os/zizhuo_vcc/work
export PYTHONPATH=/home/zizhuo/maglab_deploy/src
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT"
exec python3 -u train_magworld_h1_v4_rawsel.py \
  --signatures /nfs_beijing_os/zizhuo_vcc/signatures/h1_trainval_signatures.npz \
  --official-perts /nfs_beijing_os/zizhuo_vcc/h1_2025/pert_counts_Training.csv \
  --gene-embeddings /nfs_beijing_os/zizhuo_vcc/embeddings/vcc_gene_embeddings_v13geom_256.npy \
  --magworld-src /home/zizhuo/maglab_deploy/src \
  --model-module model_world_h1_v13 --model-class WorldModelH1V13 \
  --sparse-top-k 2048 --cheb-order 3 \
  --out "$OUT/best.pt" \
  --seed "$SEED" --holdout-mode none \
  --epochs 120 --patience 999 \
  --batch-size 24 --learning-rate 1e-4 --weight-decay 2e-4 \
  --top-k 100 \
  --lambda-global 0.2 --lambda-de 1.0 --lambda-cosine 0.5 --lambda-direction 0.15 \
  --lambda-rank 0.1 --lambda-pds 0.05 --lambda-yield 0.15 --lambda-shared-bias 0.05 \
  --yield-threshold 0.05 --yield-temperature 0.02 \
  --direction-temperature 0.05 --rank-temperature 0.05 \
  --tau 0.15 --latent-dim 64 --hidden-dim 192 \
  --magnetic-mode directed --shared-bias-mode gated --shared-bias-initial-scale 0.1 \
  --normalize-context --select-raw
