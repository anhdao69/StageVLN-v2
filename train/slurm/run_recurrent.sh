#!/usr/bin/env bash
set -euo pipefail
variant="${1:?v2 or v3}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
case "$variant" in
    v2) experiment=v2_mem64_r0 ;;
    v3) experiment=v3_mem64_r4 ;;
    *) exit 2 ;;
esac
ENV_DIR="${ENV_DIR:-/home/an221229/code/SpatialForcing-VLN/.venv}"
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/cuda-12.6.0}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
run_id="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%S)}"
run_dir="/groups/yshang/an221229/checkpoints/StageVLN-v2/${experiment}_bs64_${run_id}"
mkdir -p "$run_dir/source_snapshot"
cp -r src configs train scripts "$run_dir/source_snapshot/"
cp implementations/model_manifest.json "$run_dir/"
git rev-parse HEAD > "$run_dir/git_revision.txt"
git diff > "$run_dir/source_changes.patch"
export PYTHONPATH="$run_dir/source_snapshot/src"
"$ENV_DIR/bin/torchrun" --standalone --nproc-per-node=4 -m qwen_vl.train.run_episode \
    --config "$run_dir/source_snapshot/configs/datasets/${experiment}.json" \
    --output "$run_dir" --checkpoint-every 2500
"$ENV_DIR/bin/python" "$run_dir/source_snapshot/scripts/recurrent/upload_export.py" \
    --run "$run_dir" --repo "anhdao69/StageVLN-${experiment}-r2r"
