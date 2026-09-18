#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DATASET_CONFIG="${DATASET_CONFIG:-$PROJECT_ROOT/configs/datasets/newton_r2r_v1.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-/groups/yshang/an221229/checkpoints/StageVLN-v2/v1_r2r_sw4}"
export MAX_HISTORY_FRAMES=4
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
exec bash "$PROJECT_ROOT/train/v0_uniform8.sh"
