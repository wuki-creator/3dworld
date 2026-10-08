#!/bin/bash
# Control: v25 champion base, same seed 543. Waits for vcc-train4 to finish first.
while systemctl --user is-active vcc-train4 >/dev/null 2>&1; do sleep 30; done
cd /nfs_beijing_os/zizhuo_vcc/work
TRAINER=train_v14_aux.py bash div_train.sh 543 6 \
  /nfs_beijing_os/zizhuo_vcc/ckpts/v25_s543 \
  --model-module model_world_h1_v25 --model-class WorldModelH1V25 \
  --epochs 60 --learning-rate 1e-4 --weight-decay 1e-4 \
  --lambda-l1 1.0 --lambda-mmd 0.1 --lambda-bce 0.1 \
  --lambda-kernel 0.05 --lambda-shared-bias 0.05 --bce-temperature 0.05 \
  --graph-k 50 --freeze-base \
  --dgidb-features /nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_prior_features.npy \
  --prior-graphs /nfs_beijing_os/zizhuo_vcc/embeddings/dgidb_graph.npz,/nfs_beijing_os/zizhuo_vcc/embeddings/trrust_graph.npz \
  > /nfs_beijing_os/zizhuo_vcc/logs/v25_s543.log 2>&1
echo "control done rc=$? best=$(grep -ao '\"best_objective\": [0-9.]*' /nfs_beijing_os/zizhuo_vcc/logs/v25_s543.log | tail -1)"
cd /nfs_beijing_os/zizhuo_vcc/work
CUDA_VISIBLE_DEVICES=6 timeout 280 python3 proxy_score.py \
  --ckpt /nfs_beijing_os/zizhuo_vcc/ckpts/v25_s543/best.pt 2>/dev/null \
  | grep -aE 'raw val cosine|scale=0.50' > /nfs_beijing_os/zizhuo_vcc/work/proxy_v25_s543.txt
echo PROXY_DONE >> /nfs_beijing_os/zizhuo_vcc/work/proxy_v25_s543.txt
