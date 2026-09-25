#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_PYTHON="${EVAL_PYTHON:-/storage/anhdh35/miniconda3/envs/spatialstack_dagger/bin/python}"
R2R_DATA="${R2R_DATA:-}"
if [[ -z "$R2R_DATA" ]]; then
  R2R_DATA='/storage/anhdh35/SpatialForcing-VLN/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'
fi
MP3D_SCENES="${MP3D_SCENES:-/storage/anhdh35/SpatialForcing-VLN/data/scene_datasets}"

unset PYTHONPATH VIRTUAL_ENV
export PYTHONPATH="$ROOT/src"
export FLASH_ATTENTION_DETERMINISTIC=1
export TOKENIZERS_PARALLELISM=false

exec "$EVAL_PYTHON" -m qwen_vl.eval.habitat_r2r \
  "$ROOT/checkpoints/StageVLN-v1-r2r-sw4" \
  --policy v1_sw4 \
  --data-path "$R2R_DATA" \
  --scenes-dir "$MP3D_SCENES" \
  --output "$ROOT/evaluation/r2r_v1_sw4" \
  --max-new-tokens 32 \
  "$@"
