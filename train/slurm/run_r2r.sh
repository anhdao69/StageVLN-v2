#!/usr/bin/env bash
set -euo pipefail
variant="${1:?v0 or v1}"
export PROJECT_ROOT=/home/an221229/code/StageVLN-v2
cd "$PROJECT_ROOT"
# Explicitly selected and version-audited environment; no implicit discovery.
export ENV_DIR=/home/an221229/code/SpatialForcing-VLN/.venv
export MODEL_PATH=/groups/yshang/an221229/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DATALOADER_NUM_WORKERS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8
export DEEPSPEED_CONFIG="$PROJECT_ROOT/train/zero1.json"
export SAVE_TOTAL_LIMIT=2 SAVE_STEPS=1000 LOGGING_STEPS=10
export MAX_STEPS= MAX_SAMPLES=-1 NUM_TRAIN_EPOCHS=1
export SAVE_STRATEGY=steps SAVE_FINAL_MODEL=True GRADIENT_CHECKPOINTING=False
export LEARNING_RATE=1e-6 MM_PROJECTOR_LR=1e-5 WARMUP_STEPS=1
export ATTN_IMPLEMENTATION=flash_attention_2 SPARSE_ACTION_LOGITS=True
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
case "$variant" in
    v0) history=uniform8; export MAX_HISTORY_FRAMES=8 ;;
    v1) history=sw4; export MAX_HISTORY_FRAMES=4
        export PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=4 ;;
    *) exit 2 ;;
esac
export DATASET_CONFIG="$PROJECT_ROOT/configs/datasets/newton_r2r_${variant}.json"
export OUTPUT_DIR="/groups/yshang/an221229/checkpoints/StageVLN-v2/${variant}_r2r_${history}_bs64_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
cp "$DATASET_CONFIG" "$OUTPUT_DIR/dataset_config.json"
cp /groups/yshang/an221229/data/StageVLN-v2/data_audit.json "$OUTPUT_DIR/data_audit.json"
cp implementations/model_manifest.json "$OUTPUT_DIR/model_manifest.json"
cp train/slurm/run_r2r.sh "train/slurm/${variant}_r2r.slurm" "$OUTPUT_DIR/"
mkdir -p "$OUTPUT_DIR/source_snapshot"
cp -r src configs train scripts "$OUTPUT_DIR/source_snapshot/"
git rev-parse HEAD > "$OUTPUT_DIR/git_revision.txt"
git diff > "$OUTPUT_DIR/source_changes.patch"
"$ENV_DIR/bin/python" - "$OUTPUT_DIR" "$variant" <<'PY'
import json,os,sys,importlib.metadata as m
from pathlib import Path
manifest={'version':sys.argv[2],'base_model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','global_batch':64,'micro_batch':int(os.environ['PER_DEVICE_TRAIN_BATCH_SIZE']),'accumulation':int(os.environ['GRADIENT_ACCUMULATION_STEPS']),'world_size':4,'slurm_job_id':os.environ['SLURM_JOB_ID'],'cpus':8,'workers_per_rank':1,'loss':'mean of per-state token means; actual global accumulation-window count','tail_policy':'Accelerate even-batch padding; 631244 unique states, 631248 exposures, 4 repeats; final update has 16 exposures','packages':{x:m.version(x) for x in ['torch','transformers','accelerate','deepspeed','flash-attn']}}
assert manifest['micro_batch']*manifest['accumulation']*4==64
Path(sys.argv[1],'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
export PROJECT_ROOT="$OUTPUT_DIR/source_snapshot"
export DATASET_CONFIG="$OUTPUT_DIR/dataset_config.json"
bash "$PROJECT_ROOT/train/v0_uniform8.sh"
touch "$OUTPUT_DIR/TRAINING_COMPLETE"
"$ENV_DIR/bin/python" "$PROJECT_ROOT/scripts/train/upload_hf.py" --folder "$OUTPUT_DIR" --repo "anhdao69/StageVLN-${variant}-r2r-${history}"
