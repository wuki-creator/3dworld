#!/bin/bash
# v16 "QuantWorld" train — v13 recipe, only the field module swapped.
cd /home/zizhuo/maglab_deploy
export PYTHONPATH=/home/zizhuo/maglab_deploy/src
export CUDA_VISIBLE_DEVICES=7
mkdir -p /nfs_beijing_os/zizhuo_vcc/ckpts/v16q_seed113
exec python3 -u src/train_magworld_h1_v4.py \
  --signatures /nfs_beijing_os/zizhuo_vcc/signatures/h1_trainval_signatures.npz \
  --official-perts /home/zizhuo/vcc_data/extracted/pert_counts.csv \
  --gene-embeddings /nfs_beijing_os/zizhuo_vcc/embeddings/vcc_gene_embeddings_hybrid_256.npy \
  --magworld-src src \
  --model-module model_world_h1_v16 --model-class WorldModelH1V16 \
  --sparse-top-k 2048 --cheb-order 3 \
  --out /nfs_beijing_os/zizhuo_vcc/ckpts/v16q_seed113/magworld_h1_v16q_seed113.pt \
  --seed 113 --holdout-mode none \
  --epochs 120 --patience 999 \
  --batch-size 24 --learning-rate 1e-4 --weight-decay 2e-4 \
  --top-k 100 \
  --lambda-global 0.2 --lambda-de 1.0 --lambda-cosine 0.5 --lambda-direction 0.15 \
  --lambda-rank 0.1 --lambda-pds 0.05 --lambda-yield 0.15 --lambda-shared-bias 0.05 \
  --yield-threshold 0.05 --yield-temperature 0.02 \
  --direction-temperature 0.05 --rank-temperature 0.05 \
  --tau 0.15 --latent-dim 64 --hidden-dim 192 \
  --magnetic-mode directed --shared-bias-mode gated --shared-bias-initial-scale 0.1 \
  --normalize-context
