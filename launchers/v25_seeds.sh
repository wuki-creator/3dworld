#!/bin/bash
# 两个额外 v25 noaux 种子（s513/s514），与 norman_two 并行共用 GPU6
cd /nfs_beijing_os/zizhuo_vcc/work
COMMON="--epochs 60 --learning-rate 1e-4 --weight-decay 1e-4 --lambda-l1 1.0 --lambda-mmd 0.1 --lambda-bce 0.1 --lambda-kernel 0.05 --lambda-shared-bias 0.05 --bce-temperature 0.05 --graph-k 50 --freeze-base --dgidb-features /nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_prior_features.npy --prior-graphs /nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_graph.npz,/nfs_beijing_os/zizhuo_vcc/embeddings/trrust_graph.npz"

TRAINER=train_v14_aux.py bash div_train.sh 513 6 /nfs_beijing_os/zizhuo_vcc/ckpts/v25_noaux_s513 \
  --model-module model_world_h1_v25 --model-class WorldModelH1V25 \
  $COMMON > /nfs_beijing_os/zizhuo_vcc/logs/v25_s513.log 2>&1 &
P1=$!
TRAINER=train_v14_aux.py bash div_train.sh 514 6 /nfs_beijing_os/zizhuo_vcc/ckpts/v25_noaux_s514 \
  --model-module model_world_h1_v25 --model-class WorldModelH1V25 \
  $COMMON > /nfs_beijing_os/zizhuo_vcc/logs/v25_s514.log 2>&1 &
P2=$!
echo "launched s513=$P1 s514=$P2"
wait $P1; echo "s513 exit=$?"
wait $P2; echo "s514 exit=$?"
echo "=== SEEDS DONE $(date) ==="
