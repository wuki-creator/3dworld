#!/bin/bash
cd /home/zizhuo/maglab_deploy
export CUDA_VISIBLE_DEVICES=1
python3 -u /nfs_beijing_os/zizhuo_vcc/work/ensemble_predict.py \
  --v13 /nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed113_np1.pt \
        /nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat/magworld_h1_v13_full_seed227_np1.pt \
  --v4 /nfs_beijing_os/zizhuo_vcc/ckpts/v4_seed337 --v4-scale 0.5 \
  --mccv2 /nfs_beijing_os/zizhuo_vcc/ckpts/mccv2_h1ft_GBM8.pt \
          /nfs_beijing_os/zizhuo_vcc/ckpts/mccv2_h1ft_A172.pt \
  --controls-dir /home/zizhuo/vcc_data/extracted \
  --genes /home/zizhuo/vcc_data/extracted/gene_names.csv \
  --perts /home/zizhuo/vcc_data/extracted/pert_counts.csv \
  --val-signatures /nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz \
  --out /nfs_beijing_os/zizhuo_vcc/preds/ensemble_submission.h5ad \
  > /nfs_beijing_os/zizhuo_vcc/logs/ensemble.log 2>&1
