#!/usr/bin/env bash
set -euo pipefail

# ====== OUTPUT CSV ======
CSV="/EDITS/tap_runs/sweeps/full_grid.csv"
mkdir -p "$(dirname "$CSV")"
rm -f "$CSV"  # fresh file; comment this out if you want to append on re-runs

# ====== CONSTANTS (NOT SWEPT) ======
EPOCHS=105                       # fixed as requested
GPU="0"
INP="/EDITS/021221_C16-1_8bit_PFFC-BrightAdj.tif"
OUTDIR="/EDITS/tap_runs"

BASE="python 01_fine-tune.py \
  --input_train $INP \
  --input_val   $INP \
  --split_train 0 0.8 \
  --split_val   0.8 1.0 \
  --outdir $OUTDIR \
  --epochs $EPOCHS \
  --gpu $GPU \
  --tensorboard True \
  --metrics_csv $CSV \
  --sweep_tag fullgrid"

# ====== SWEEP SETS ======
SEEDS=(42 1337 7 21 101 2024) # can skip this  

BACKBONES=(unet) # can skip this                                  
PROJHEADS=(minimal_batchnorm minimal)            
CLASSHEADS=(minimal) # can skip this                             
PERM_EQUIV=(True False) # can skip this                           
FEATURES=(16 32 48 64 80 96)                      

BATCHS=(16 24 32 64 96 128)                      
SIZES=(16 32 48 64 96 128)                       
CAM_SIZES=(64 96 128 256 512 1024)                                     
TRAIN_SAMPLES=(500 10000 20000 50000 75000 100000)
VAL_SAMPLES=(500 1000 2000 5000 10000 15000)
CHANNELS=(0 1) # can skip this                                    
REJECT_BG=(False True)                           
CAM_SUBS=(1 2 3 4 5 6)
WRITE_CAMS=(False) # can skip this                                

AUGS=(0 1 2 3 4 5)
SUBSAMPLE=(1 2 3 4 6 8)
DELTAS=(0.5 1 2 3 4 5)

FRAMES=(2 3 4 5 6 8)

LRs=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3)
LR_SCHEDS=(cyclic plateau constant)              
LR_PATIENCE=(10 20 25 35 50 75)

BINARIZE=(False True)                             
DECOR_LOSSES=(0.0 0.001 0.005 0.01 0.02 0.05)
SAVE_EVERY=(10 20 25 50 100 200)
NUM_WORKERS=(0 2 4 8 12 16)
VIZ_FREQ=(1 5 10 20 50 100)

# ====== GRID LOOP ======
for SEED in "${SEEDS[@]}"; do
for BB in "${BACKBONES[@]}"; do
for PH in "${PROJHEADS[@]}"; do
for CH in "${CLASSHEADS[@]}"; do
for PE in "${PERM_EQUIV[@]}"; do
for FEAT in "${FEATURES[@]}"; do
for BS in "${BATCHS[@]}"; do
for SZ in "${SIZES[@]}"; do
  # optional CAM size; if none set, pass nothing
  for CAM in "${CAM_SIZES[@]:-__NONE__}"; do
for TRS in "${TRAIN_SAMPLES[@]}"; do
for VLS in "${VAL_SAMPLES[@]}"; do
for C in "${CHANNELS[@]}"; do
for RBG in "${REJECT_BG[@]}"; do
for CS in "${CAM_SUBS[@]}"; do
for WFC in "${WRITE_CAMS[@]}"; do
for AUG in "${AUGS[@]}"; do
for SUB in "${SUBSAMPLE[@]}"; do
for D in "${DELTAS[@]}"; do
for FR in "${FRAMES[@]}"; do
for LR in "${LRs[@]}"; do
for LRS in "${LR_SCHEDS[@]}"; do
for LRP in "${LR_PATIENCE[@]}"; do
for BIN in "${BINARIZE[@]}"; do
for DL in "${DECOR_LOSSES[@]}"; do
for SCE in "${SAVE_EVERY[@]}"; do
for NW in "${NUM_WORKERS[@]}"; do
for VF in "${VIZ_FREQ[@]}"; do

  NAME="bb${BB}_ph${PH}_ch${CH}_pe${PE}_f${FEAT}_bs${BS}_sz${SZ}"
  if [[ "${CAM:-__NONE__}" != "__NONE__" ]]; then
    NAME+="_cams${CAM}"
  fi
  NAME+="_tr${TRS}_vl${VLS}_chn${C}_rbg${RBG}_cs${CS}_wfc${WFC}_aug${AUG}_sub${SUB}_d${D}_fr${FR}"
  NAME+="_lr${LR}_lrs${LRS}_lrp${LRP}_bin${BIN}_dl${DL}_sce${SCE}_nw${NW}_vf${VF}_seed${SEED}"

  echo ">>> Running $NAME"
  CMD="$BASE \
    --name $NAME \
    --seed $SEED \
    --backbone $BB \
    --projhead $PH \
    --classhead $CH \
    --perm_equiv $PE \
    --features $FEAT \
    --batchsize $BS \
    --size $SZ \
    --train_samples_per_epoch $TRS \
    --val_samples_per_epoch $VLS \
    --channels $C \
    --reject_background $RBG \
    --cam_subsampling $CS \
    --write_final_cams $WFC \
    --augment $AUG \
    --subsample $SUB \
    --delta $D \
    --frames $FR \
    --lr $LR \
    --lr_scheduler $LRS \
    --lr_patience $LRP \
    --binarize $BIN \
    --decor_loss $DL \
    --save_checkpoint_every $SCE \
    --num_workers $NW \
    --visual_dataset_frequency $VF"

  if [[ "${CAM:-__NONE__}" != "__NONE__" ]]; then
    CMD+=" --cam_size $CAM"
  fi

  CUDA_VISIBLE_DEVICES=$GPU bash -lc "$CMD"

done; done; done; done; done; done; done; done; done; done; done; done
done; done; done; done; done; done; done; done; done; done; done; done
done; done; done; done
