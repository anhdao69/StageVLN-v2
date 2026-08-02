#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# To train another JanusVLN annotation, point DATASET_CONFIG at a JSON file
# containing annotation_path and data_path in SpatialStack's dataset format.
export DATASET_CONFIG="${DATASET_CONFIG:-$PROJECT_ROOT/configs/datasets/janusvln_r2r_rxr.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/spatial_forcing_vln_r2r_rxr_epoch1}"

# Full one-epoch recipe: all records, explicit VGGT positional encoding, and
# step-based checkpoints. MAX_STEPS is deliberately not passed.
export NUM_TRAIN_EPOCHS=1
export MAX_SAMPLES=-1
export SF_USE_VGGT_PE=True
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
unset MAX_STEPS

exec bash "$PROJECT_ROOT/scripts/train/train_spatial_forcing_vln.sh"
