#!/bin/bash
# Formal 60-epoch trainings for v34-v37, GPU6, sequential, same seed 543.
cd /nfs_beijing_os/zizhuo_vcc/work
for spec in "v34 model_world_h1_v34 WorldModelH1V34" \
            "v35 model_world_h1_v35 WorldModelH1V35" \
            "v36 model_world_h1_v36 WorldModelH1V36" \
            "v37 model_world_h1_v37 WorldModelH1V37"; do
  set -- $spec
  name=$1; mod=$2; cls=$3
  echo "=== TRAIN $name $(date +%H:%M:%S) ==="
  TRAINER=train_v14_aux.py bash div_train.sh 543 6 \
    /nfs_beijing_os/zizhuo_vcc/ckpts/${name}_s543 \
    --model-module $mod --model-class $cls \
    --epochs 60 --learning-rate 1e-4 --weight-decay 1e-4 \
    --lambda-l1 1.0 --lambda-mmd 0.1 --lambda-bce 0.1 \
    --lambda-kernel 0.05 --lambda-shared-bias 0.05 --bce-temperature 0.05 \
    --graph-k 50 --freeze-base \
    --dgidb-features /nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_prior_features.npy \
    --prior-graphs /nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_graph.npz,/nfs_beijing_os/zizhuo_vcc/embeddings/trrust_graph.npz \
    > /nfs_beijing_os/zizhuo_vcc/logs/${name}_s543.log 2>&1
  echo "--- done $name rc=$? best=$(grep -ao '"best_objective": [0-9.]*' /nfs_beijing_os/zizhuo_vcc/logs/${name}_s543.log | tail -1)"
done
echo ALL_TRAIN_DONE
