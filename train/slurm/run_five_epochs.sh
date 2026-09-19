#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:?PROJECT_ROOT required}"
ENV_DIR="${ENV_DIR:-/home/an221229/code/SpatialForcing-VLN/.venv}"
export CUDA_HOME=/apps/cuda/cuda-12.6.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_XET_HIGH_PERFORMANCE=0
run_dir="${RUN_DIR:-/groups/yshang/an221229/checkpoints/StageVLN-v2/re_mem_${SLURM_JOB_ID:?}}"
resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    [[ -f "$RESUME_CHECKPOINT/manifest.json" && -d "$run_dir/source_snapshot" ]]
    resume_args=(--resume "$RESUME_CHECKPOINT")
else
    [[ ! -e "$run_dir" ]] || { echo 'Run directory exists; use explicit resume'; exit 1; }
    mkdir -p "$run_dir/source_snapshot"
    cp -r src configs train scripts "$run_dir/source_snapshot/"
    if [[ -f SOURCE_REVISION ]]; then
        cp SOURCE_REVISION "$run_dir/git_revision.txt"
    else
        git rev-parse HEAD > "$run_dir/git_revision.txt"
        git diff > "$run_dir/source_changes.patch"
    fi
fi
rm -f "$run_dir/TRAINING_PROCESS_EXITED"
export PYTHONPATH="$run_dir/source_snapshot/src"
"$ENV_DIR/bin/python" "$run_dir/source_snapshot/scripts/recurrent/upload_epochs.py" \
    --run "$run_dir" --prefix anhdao69/StageVLN-v2_mem64_r0-r2r > "$run_dir/uploads.log" 2>&1 &
uploader_pid=$!
trap 'kill "$uploader_pid" 2>/dev/null || true' EXIT
set +e
"$ENV_DIR/bin/torchrun" --standalone --nproc-per-node=4 -m qwen_vl.train.run_episode \
    --config "$run_dir/source_snapshot/configs/datasets/v2_mem64_r0_5epochs.json" \
    --output "$run_dir" --checkpoint-every 500 "${resume_args[@]}"
training_status=$?
touch "$run_dir/TRAINING_PROCESS_EXITED"
wait "$uploader_pid"
upload_status=$?
set -e
if [[ "$training_status" != 0 ]]; then exit "$training_status"; fi
exit "$upload_status"
