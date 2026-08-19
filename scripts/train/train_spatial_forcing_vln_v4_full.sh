#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Full controlled run matched to train_spatial_forcing_vln_full.sh: one epoch
# over the combined R2R + RxR annotations with only v4 depth supervision added.
export DATASET_CONFIG="${DATASET_CONFIG:-$PROJECT_ROOT/configs/datasets/janusvln_r2r_rxr.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/spatial_forcing_vln_v4_depth_r2r_rxr_epoch1}"
export NUM_TRAIN_EPOCHS=1
export MAX_SAMPLES=-1
export SF_USE_VGGT_PE=True
export DEPTH_SUPERVISION_ENABLED=True
export DEPTH_LOSS_WEIGHT="${DEPTH_LOSS_WEIGHT:-0.05}"
export DEPTH_HEAD_LR="${DEPTH_HEAD_LR:-1e-5}"
export DEPTH_OUTLIER_KEEP_RATIO="${DEPTH_OUTLIER_KEEP_RATIO:-0.98}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
unset MAX_STEPS

exec bash "$PROJECT_ROOT/scripts/train/train_spatial_forcing_vln_v4.sh"
