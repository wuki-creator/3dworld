#!/bin/bash
# Norman 混训两路（修复 aux 基因宽度 bug + pickle 协议后重启），GPU6 顺序执行
cd /nfs_beijing_os/zizhuo_vcc/work
COMMON="--epochs 60 --learning-rate 1e-4 --weight-decay 1e-4 --lambda-l1 1.0 --lambda-mmd 0.1 --lambda-bce 0.1 --lambda-kernel 0.05 --lambda-shared-bias 0.05 --bce-temperature 0.05 --graph-k 50 --freeze-base --dgidb-features /nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_prior_features.npy"

echo "=== RUN1 v25n s521 $(date) ==="
TRAINER=train_v14_aux.py bash div_train.sh 521 6 /nfs_beijing_os/zizhuo_vcc/ckpts/v25_norman_s521 \
  --model-module model_world_h1_v25 --model-class WorldModelH1V25 \
  $COMMON \
  --prior-graphs /nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_graph.npz,/nfs_beijing_os/zizhuo_vcc/embeddings/trrust_graph.npz \
  --aux-signatures /nfs_beijing_os/zizhuo_vcc/pertseq/norman/norman_aux.npz --aux-weight 0.2 \
  > /nfs_beijing_os/zizhuo_vcc/logs/norman_521.log 2>&1
echo "RUN1 exit=$?"

echo "=== RUN2 v18n s522 $(date) ==="
TRAINER=train_v14_aux.py bash div_train.sh 522 6 /nfs_beijing_os/zizhuo_vcc/ckpts/v18_norman_s522 \
  --model-module model_world_h1_v18 --model-class WorldModelH1V18 \
  $COMMON \
  --aux-signatures /nfs_beijing_os/zizhuo_vcc/pertseq/norman/norman_aux.npz --aux-weight 0.2 \
  > /nfs_beijing_os/zizhuo_vcc/logs/norman_522.log 2>&1
echo "RUN2 exit=$?"
echo "=== ALL DONE $(date) ==="
