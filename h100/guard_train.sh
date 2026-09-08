#!/usr/bin/env bash
# Idle-guarded finetune loop: trains only when no other process holds the GPU.
# Usage: guard_train.sh <workdir> <target_epochs> <tag> <warm_start_pth> [lr] [batch] [seed]
set -u
W=$1; TARGET=$2; TAG=$3; WARM0=$4; LR=${5:-3e-5}; BATCH=${6:-32}; SEED=${7:-777}
REPO=$W/sp/repo; OUT=$REPO/weights/unet_transformer/split_0; LOG=$W/logs/train_$TAG.log
PY=$W/venv/bin/python
export PYTHONPATH=$REPO/src KAGGLE_CONFIG_DIR=$W/kg
export BIOHUB_STOP_FILE=$W/STOP BIOHUB_TRAIN_SEED=$SEED BIOHUB_DIV_LOSS_WEIGHT=${DIVW:-1.0} BIOHUB_TRAIN_TIME_CAP_H=0 BIOHUB_YIELD_CHECK=1
mkdir -p $OUT $W/wts/$TAG

idle_for() { # true if no foreign compute app for $1 consecutive 15s samples
  local n=$1; local i
  for ((i=0;i<n;i++)); do
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; then return 1; fi
    sleep 15
  done; return 0
}
done_epochs() { [ -f $OUT/epochs_done.txt ] && wc -l < $OUT/epochs_done.txt || echo 0; }

echo "[guard $(date -u +%H:%M:%S)] target=$TARGET tag=$TAG warm=$WARM0 lr=$LR batch=$BATCH" >> $LOG
while true; do
  D=$(done_epochs)
  if [ -f $W/STOP ]; then echo "[guard] STOP file present -> exiting" >> $LOG; break; fi
  if [ "$D" -ge "$TARGET" ]; then echo "[guard] target reached ($D epochs)" >> $LOG; break; fi
  if idle_for 6; then
    if [ -f $OUT/edge_predictor_last.pth ]; then export BIOHUB_WARM_START=$OUT/edge_predictor_last.pth; else export BIOHUB_WARM_START=$WARM0; fi
    REM=$((TARGET - D))
    echo "[guard $(date -u +%H:%M:%S)] GPU idle -> training $REM more epoch(s) from $BIOHUB_WARM_START" >> $LOG
    cd $REPO && nice -n 5 $PY scripts/train_unet_transformer.py --data-dir $W/data/train --splits $W/ft_splits.json --split 0 \
       --epochs $REM --lr $LR --batch-size $BATCH --num-workers 6 --single-gpu >> $LOG 2>&1
    echo "[guard $(date -u +%H:%M:%S)] trainer exited rc=$?" >> $LOG
    cp -f $OUT/edge_predictor_last.pth $W/wts/$TAG/ 2>/dev/null; cp -f $OUT/edge_predictor_best.pth $W/wts/$TAG/ 2>/dev/null
    cp -f $OUT/config.json $OUT/epochs_done.txt $W/wts/$TAG/ 2>/dev/null
  else
    sleep 45
  fi
done
echo "[guard] DONE $TAG" >> $LOG
