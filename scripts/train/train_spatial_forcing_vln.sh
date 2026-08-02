#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-facebook/VGGT-1B}"
JANUSVLN_DATA_ROOT="${JANUSVLN_DATA_ROOT:-$PROJECT_ROOT/data/JanusVLN_data}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/spatial_forcing_vln_r2r}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SF_PROJECTOR_LR="${SF_PROJECTOR_LR:-1e-5}"
SF_LOSS_WEIGHT="${SF_LOSS_WEIGHT:-0.3}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"

# Python multiprocessing cleanup is unreliable on some shared Lustre mounts.
# Keep worker sockets and Triton compilation artifacts on node-local storage.
LOCAL_TMPDIR="${SPATIAL_FORCING_TMPDIR:-${SLURM_TMPDIR:-/tmp/$USER/spatial-forcing-${SLURM_JOB_ID:-manual}}}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$LOCAL_TMPDIR"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
# huggingface_hub deprecated this knob; do not emit one warning per rank.
unset HF_HUB_ENABLE_HF_TRANSFER
export TMPDIR="$LOCAL_TMPDIR"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LOCAL_TMPDIR/triton-cache}"
mkdir -p "$TRITON_CACHE_DIR"

# The Triton 3.6 build pulled by torch 2.10 is rejected by FLA's Qwen3.5
# gated-delta backward on Hopper.  Fail before allocating four model replicas
# and give the exact uv command used by the verified environment.
"$PYTHON_BIN" - <<'PY'
from packaging.version import Version
import torch
import triton
import sys

version = Version(triton.__version__)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
    if Version("3.4") <= version < Version("3.7.1"):
        raise SystemExit(
            "Spatial Forcing on Hopper requires Triton >=3.7.1; run "
            f"`uv pip install --python {sys.executable} --no-deps triton==3.7.1`."
        )
PY

train_args=(
    --model_name_or_path "$MODEL_PATH"
    --geometry_encoder_path "$TEACHER_MODEL_PATH"
    --spatial_forcing_enabled True
    --use_geometry_encoder False
    --use_geometry_fusion False
    --sf_student_layer 24
    --sf_teacher_layer 23
    --sf_projector_hidden_dim 4096
    --sf_loss_weight "$SF_LOSS_WEIGHT"
    --sf_verify_invariants True
    --tune_mm_llm True
    --tune_mm_mlp True
    --tune_mm_vision False
    --dataset_use janusvln_r2r
    --janusvln_data_root "$JANUSVLN_DATA_ROOT"
    --max_samples "$MAX_SAMPLES"
    --shuffle True
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    --bf16
    --tf32 True
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LEARNING_RATE"
    --mm_projector_lr "$SF_PROJECTOR_LR"
    --vision_tower_lr 0
    --optim adamw_torch
    --model_max_length 12800
    --data_flatten False
    --max_pixels $((576*28*28))
    --min_pixels $((16*28*28))
    --num_train_epochs 1
    --max_steps "$MAX_STEPS"
    --warmup_steps "$WARMUP_STEPS"
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --logging_steps "$LOGGING_STEPS"
    --save_strategy "$SAVE_STRATEGY"
    --save_steps 1000
    --save_total_limit 1
    --deepspeed "$PROJECT_ROOT/scripts/zero2_opt.json"
    --gradient_checkpointing
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --group_by_modality_length True
    --seed 42
    --report_to none
)

"$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    src/qwen_vl/train/train_qwen.py \
    "${train_args[@]}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
