#!/bin/bash
# Orchestrate: wait vdiv runs -> proxy eval each -> ensemble h5ad + prep -> update timer script.
set -x
W=/nfs_beijing_os/zizhuo_vcc/work
K=/nfs_beijing_os/zizhuo_vcc/ckpts
P=/nfs_beijing_os/zizhuo_vcc/preds
L=/nfs_beijing_os/zizhuo_vcc/logs
EXTRACT=/nfs_beijing/zizhuo/systemdisk_backup/vcc_data/extracted
R=$L/vdiv_results.txt
: > $R

# 1. wait for the three 60-epoch runs (434 is 150ep, do not block on it)
for i in $(seq 1 150); do
  N=$(systemctl --user list-units 'vdiv-431' 'vdiv-433' 'vdiv-435' --no-legend 2>/dev/null | grep -c running)
  echo "poll $i: $N running" >> $R
  [ "$N" -eq 0 ] && break
  sleep 60
done

# 2. proxy eval each diversity ckpt
cd $W
for d in v14d_unfreeze3e5 v14d_lossmix v14d_long150 v14d_graphk100; do
  CK=$K/$d/best.pt
  if [ -f "$CK" ]; then
    R2=$(CUDA_VISIBLE_DEVICES=3 python3 proxy_score.py --ckpt "$CK" 2>/dev/null | grep -E 'raw val cosine|scale=0.50' | tr '\n' ' ')
    echo "div=$d $R2" >> $R
  else
    echo "div=$d MISSING (not finished yet)" >> $R
  fi
done

# 3. pick members: the 6 known sharp + any diversity ckpt with raw cos >= 0.38
MEMBERS="$K/v14g_full/best.pt $K/v14g_s421/best.pt $K/v14g_s422/best.pt $K/v14g_s423/best.pt $K/v13_compat/magworld_h1_v13_full_seed113_np1.pt $K/v15_full/best.pt"
for d in v14d_unfreeze3e5 v14d_lossmix v14d_long150 v14d_graphk100; do
  CK=$K/$d/best.pt
  [ -f "$CK" ] || continue
  C=$(grep "div=$d " $R | grep -oE 'raw val cosine=[0-9.]+' | grep -oE '[0-9.]+$')
  if [ -n "$C" ]; then
    OK=$(python3 -c "print(1 if float('$C')>=0.38 else 0)")
    [ "$OK" = "1" ] && MEMBERS="$MEMBERS $CK"
  fi
done
echo "MEMBERS=$MEMBERS" >> $R

# 4. ensemble h5ad (tight decode, amp 1.0, clip 3.0) then vcc prep
export CUDA_VISIBLE_DEVICES=3
export TMPDIR=$W/tmp
mkdir -p $TMPDIR
python3 -u $W/ensemble_predict_h1.py \
  --ckpts $MEMBERS \
  --top-k -1 --scale 1.0 --self-scale 1.0 --amp 1.0 --clip 3.0 \
  --decode-style tight --jitter-shape 200 --cells-per-target 400 \
  --controls-dir $EXTRACT --genes $EXTRACT/gene_names.csv --perts $EXTRACT/pert_counts.csv \
  --val-signatures /nfs_beijing_os/zizhuo_vcc/signatures/h1_val_signatures.npz \
  --out $P/ens6_raw_amp1.0_c3.h5ad >> $R 2>&1

/home/zizhuo/.local/bin/vcc prep $P/ens6_raw_amp1.0_c3.h5ad \
  -g $EXTRACT/gene_names.csv --perts $EXTRACT/pert_counts.csv \
  -o $P/ens6_raw_amp1.0_c3.vcc >> $R 2>&1

ls -la $P/ens6_raw_amp1.0_c3.vcc >> $R 2>&1
echo "=== orchestrator done $(date) ===" >> $R
