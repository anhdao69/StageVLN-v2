#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# v4 is the controlled v0 + current-frame VGGT pseudo-depth ablation.  This
# launcher defaults to R2R and remains suitable for bounded smoke runs through
# MAX_STEPS/MAX_SAMPLES overrides.
export DEPTH_SUPERVISION_ENABLED=True
export SF_USE_VGGT_PE="${SF_USE_VGGT_PE:-True}"
export DEPTH_LOSS_WEIGHT="${DEPTH_LOSS_WEIGHT:-0.05}"
export DEPTH_HEAD_LR="${DEPTH_HEAD_LR:-1e-5}"
export DEPTH_OUTLIER_KEEP_RATIO="${DEPTH_OUTLIER_KEEP_RATIO:-0.98}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/spatial_forcing_vln_v4_depth_r2r}"

exec bash "$PROJECT_ROOT/scripts/train/train_spatial_forcing_vln.sh"
