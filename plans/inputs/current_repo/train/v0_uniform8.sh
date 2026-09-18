#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

ENV_DIR="${ENV_DIR:-$PROJECT_ROOT/../SpatialForcing-VLN/.venv}"
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    echo "Missing uv environment: $ENV_DIR" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# DeepSpeed checks for a CUDA toolkit at import time. Prefer an already loaded
# toolkit, then derive it from nvcc, with the local H100 cluster as a fallback.
if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    elif [[ -x /apps/cuda/cuda-12.6.0/bin/nvcc ]]; then
        CUDA_HOME=/apps/cuda/cuda-12.6.0
    else
        echo "CUDA_HOME is unset and nvcc was not found; load a CUDA toolkit" >&2
        exit 1
    fi
    export CUDA_HOME
fi
export PATH="$CUDA_HOME/bin:$PATH"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DATASET_CONFIG="${DATASET_CONFIG:-$PROJECT_ROOT/configs/datasets/newton_r2r_uniform8.json}"
default_cache_dir="$PROJECT_ROOT/cache"
if [[ -d /lustre/fs1/groups/yshang/an221229/cache/huggingface/hub ]]; then
    default_cache_dir=/lustre/fs1/groups/yshang/an221229/cache/huggingface/hub
fi
CACHE_DIR="${CACHE_DIR:-${HF_HUB_CACHE:-$default_cache_dir}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/v0_uniform8}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
allocated_cpus="${SLURM_CPUS_PER_TASK:-$((NPROC_PER_NODE * 5))}"
default_workers=$((allocated_cpus / NPROC_PER_NODE - 1))
(( default_workers < 0 )) && default_workers=0
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-$default_workers}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-True}"

LOCAL_TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/qwen35-sft-${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$LOCAL_TMPDIR/triton-cache"
export TMPDIR="$LOCAL_TMPDIR"
export TRITON_CACHE_DIR="$LOCAL_TMPDIR/triton-cache"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python - "$DATASET_CONFIG" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
for key in ("annotation_path", "data_path"):
    path = os.path.expandvars(os.path.expanduser(config[key]))
    predicate = os.path.isfile if key == "annotation_path" else os.path.isdir
    if not predicate(path):
        raise SystemExit(f"Dataset {key} does not exist: {path}")
print(f">>>>> dataset={config['dataset_name']}")
print(f">>>>> annotation={config['annotation_path']}")
print(f">>>>> data_root={config['data_path']}")
PY

train_args=(
    --model_name_or_path "$MODEL_PATH"
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --tune_mm_llm True
    --tune_mm_mlp True
    --tune_mm_vision False
    --dataset_config "$DATASET_CONFIG"
    --max_history_frames 8
    --max_samples "$MAX_SAMPLES"
    --shuffle True
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    --bf16 True
    --tf32 True
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --mm_projector_lr "$MM_PROJECTOR_LR"
    --optim adamw_torch
    --model_max_length 12800
    --max_pixels $((576*28*28))
    --min_pixels $((16*28*28))
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --lr_scheduler_type cosine
    --warmup_steps "$WARMUP_STEPS"
    --weight_decay 0.01
    --logging_steps "$LOGGING_STEPS"
    --save_strategy "$SAVE_STRATEGY"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --save_final_model "$SAVE_FINAL_MODEL"
    --deepspeed "$PROJECT_ROOT/train/zero2.json"
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --group_by_modality_length True
    --ddp_find_unused_parameters False
    --seed 42
    --report_to none
)

if (( DATALOADER_NUM_WORKERS > 0 )); then
    train_args+=(--dataloader_persistent_workers True --dataloader_prefetch_factor 4)
fi
if [[ -n "$MAX_STEPS" ]]; then
    if ! [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_STEPS must be a positive integer" >&2
        exit 2
    fi
    train_args+=(--max_steps "$MAX_STEPS")
fi

echo ">>>>> Qwen3.5-4B plain SFT on $NPROC_PER_NODE GPUs"
echo ">>>>> gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS gradient_checkpointing=$GRADIENT_CHECKPOINTING"
start_seconds=$SECONDS
python -m torch.distributed.run \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m qwen_vl.train.train_qwen \
    "${train_args[@]}" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
elapsed_seconds=$((SECONDS - start_seconds))
echo ">>>>> end-to-end wall time: ${elapsed_seconds}s" | tee -a "$OUTPUT_DIR/train.log"
