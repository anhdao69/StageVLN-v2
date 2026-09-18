# Reproducing v2/v3 validation

Run from the `recurrent_memory` checkout root, inside an allocation with four
H100 80GB GPUs. The shared environment has another project's editable install,
so setting `PYTHONPATH` is required. These commands are documentation; no job
submission is performed by writing this file. Use a fresh output directory for
every run. Paths below name the recorded artifacts.

```bash
export ENV_DIR=/home/an221229/code/SpatialForcing-VLN/.venv
export PYTHONPATH="$PWD/src"
export CUDA_HOME=/apps/cuda/cuda-12.6.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export FLASH_ATTENTION_DETERMINISTIC=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CHECKPOINT_ROOT=/groups/yshang/an221229/checkpoints/StageVLN-v2

"$ENV_DIR/bin/python" -m pytest -q tests
"$ENV_DIR/bin/python" scripts/recurrent/check_adapter.py
"$ENV_DIR/bin/python" scripts/recurrent/check_temporal_parity.py
"$ENV_DIR/bin/python" scripts/recurrent/check_resume.py
"$ENV_DIR/bin/python" scripts/recurrent/check_delayed_cue.py
"$ENV_DIR/bin/python" scripts/recurrent/check_real_episode_overfit.py
"$ENV_DIR/bin/python" scripts/recurrent/check_full_adapter.py
"$ENV_DIR/bin/python" scripts/recurrent/check_full_temporal.py
```

The scripts contain explicit tiny-model or real-model assertions; inspect their
documented constants before changing a diagnostic's scope. Run GPU commands
sequentially so another model copy does not distort peaks or cause OOM.

## Recorded four-GPU production-shape smoke

```bash
"$ENV_DIR/bin/torchrun" --standalone --nproc-per-node=4 \
  -m qwen_vl.train.run_episode --config configs/datasets/v2_mem64_r0.json \
  --output "$CHECKPOINT_ROOT/verify_v2_recurrent" \
  --max-episodes 64 --max-updates 32 --checkpoint-every 0

"$ENV_DIR/bin/torchrun" --standalone --nproc-per-node=4 \
  -m qwen_vl.train.run_episode --config configs/datasets/v3_mem64_r4.json --reader-microbatch 4 \
  --output "$CHECKPOINT_ROOT/verify_v3_recurrent" \
  --max-episodes 64 --max-updates 32 --checkpoint-every 0
```

Each consumes 2,048 labeled states, saves update 32 plus a full policy export,
and writes `SMOKE_COMPLETE`. The subset contains complete episodes; stopping at
update 32 leaves live cursors for a resume test. The learning-rate schedule uses
the selected 64-episode subset's epoch length, so these losses are smoke
measurements, not a 32-step prefix of the full-epoch learning-rate schedule.
The saved v3 update 32 was produced with reader batch 4. Its original config lives
in that run's source snapshot. The selected production config now uses batch 8;
the explicit override above reproduces the earlier execution choice, while its
new config manifest is not interchangeable with that original checkpoint.

The final batch 8 timing candidate used the same v3 command with
`--reader-microbatch 8 --skip-final-save`, output
`$CHECKPOINT_ROOT/verify_v3_b8_deterministic`. The extra K8 gradient check is:

```bash
"$ENV_DIR/bin/python" scripts/recurrent/check_temporal_parity.py --reader-microbatch 8
```

The matched chronological controls use exactly the same bounded label budget:

```bash
for variant in v1_current_ep v1_sw4_ep; do
  "$ENV_DIR/bin/torchrun" --standalone --nproc-per-node=4 \
    -m qwen_vl.train.run_episode --config "configs/datasets/${variant}.json" \
    --output "$CHECKPOINT_ROOT/verify_${variant}" \
    --max-episodes 64 --max-updates 32 --checkpoint-every 0 --skip-final-save
done
```

The longest-instruction stress configs under `implementations/` pin a separately
audited 16-episode manifest (1,165 states). Their final versions match the selected
reader batches4/8 and checkpointing off/on for v2/v3, respectively.
Use `--max-updates 30 --checkpoint-every 0 --skip-final-save`; the scheduler
finishes after 19 updates including its13-label tail. Stress timing excludes the
partial tail when extrapolating to the canonical 9,864-update epoch.

## Resume, saved weights, actor, and trace

Strict resume requires the original source/config/runtime manifest. The recorded
v3 continuation used its saved source snapshot, because diagnostic source files
were added later and source fingerprint changes are deliberately rejected.

```bash
PYTHONPATH="$CHECKPOINT_ROOT/verify_v3_recurrent/source_snapshot/src" \
"$ENV_DIR/bin/torchrun" --standalone --nproc-per-node=4 \
  -m qwen_vl.train.run_episode \
  --config "$CHECKPOINT_ROOT/verify_v3_recurrent/source_snapshot/configs/datasets/v3_mem64_r4.json" \
  --output "$CHECKPOINT_ROOT/verify_v3_resumed" \
  --resume "$CHECKPOINT_ROOT/verify_v3_recurrent/checkpoint-32" \
  --max-episodes 64 --max-updates 33 --checkpoint-every 0 --skip-final-save

"$ENV_DIR/bin/python" scripts/recurrent/check_export.py \
  --run "$CHECKPOINT_ROOT/verify_v3_recurrent" --step 32 \
  --output implementations/recurrent_export_check.json
"$ENV_DIR/bin/python" scripts/recurrent/record_trace.py \
  --export "$CHECKPOINT_ROOT/verify_v3_recurrent/export" \
  --output implementations/recurrent_reference_trace.json
```

For a later full run, launch the versioned wrapper from the branch root after
creating `slurm_logs/`. It snapshots source, uses global64, K8, four GPUs/eight
CPUs, checkpoints every 2,500 updates, and uploads the completed export. No
submission command has been executed during this implementation task.

## Original IID v0/v1 timing commands

These use the main baseline implementation and a shared 2,048-state subset
sampled uniformly from the canonical dataset. Its exact indices, annotation
paths, configs, and hashes are in `implementations/baseline_timing_subset.json`.
The prepared subset/config files remain under the groups data directory.

```bash
export PROJECT_ROOT=/home/an221229/code/StageVLN-v2
export ENV_DIR=/home/an221229/code/SpatialForcing-VLN/.venv
export MODEL_PATH=/groups/yshang/an221229/cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NPROC_PER_NODE=4 DATALOADER_NUM_WORKERS=1
export MAX_STEPS=32 MAX_SAMPLES=-1 SAVE_STRATEGY=no SAVE_FINAL_MODEL=False
export GRADIENT_CHECKPOINTING=False SKIP_MEMORY_METRICS=False LOGGING_STEPS=1
export FLASH_ATTENTION_DETERMINISTIC=0 MASTER_PORT=29731

export PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8 MAX_HISTORY_FRAMES=8
export DATASET_CONFIG=/groups/yshang/an221229/data/StageVLN-v2/timing_v0_config.json
export OUTPUT_DIR=/groups/yshang/an221229/checkpoints/StageVLN-v2/benchmark_v0_timing32
bash "$PROJECT_ROOT/train/v0_uniform8.sh"

export PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=4 MAX_HISTORY_FRAMES=4
export DATASET_CONFIG=/groups/yshang/an221229/data/StageVLN-v2/timing_v1_config.json
export OUTPUT_DIR=/groups/yshang/an221229/checkpoints/StageVLN-v2/benchmark_v1_timing32
bash "$PROJECT_ROOT/train/v1_sw4.sh"
```

Use fresh output directories when repeating these commands. These call the
bounded SFT entry points directly, without a Slurm submission or upload helper.
Recorded source was `7a76695`; the main follow-up `c340227` changes only the
prepared v0 walltime. Original raw training logs remain in the run directories;
three committed logs have whitespace-only normalization recorded in
`implementations/log_normalization.json`.
