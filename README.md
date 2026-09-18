# Qwen3.5 R2R training: v0–v3

The `recurrent_memory` branch adds v2 (Memory64 + current) and v3 (Memory64 +
Recent4 + current). Read the [detailed implementation report](reports/v2_v3_implementation_report.md),
[acceptance record](reports/acceptance.md), and [validation commands](reports/validation_commands.md)
for architecture, measured throughput, correctness evidence, and remaining evaluation work.
Measured one-epoch estimates on four H100s are about22 h for v0,9 h for v1,
5.9 h for v2, and9.4 h for v3; see [the timing report](reports/training_time_estimates.md)
for settings, overheads, and uncertainty.

Recurrent configs are `configs/datasets/v2_mem64_r0.json` and `v3_mem64_r4.json`.
Both use K8, global batch64, FP32 trainable/state/Adam storage,
BF16 compute, deterministic FA2, and native optimizer-state sharding on four H100s.
v2 uses reader microbatch4 without activation checkpointing; v3 uses
reader microbatch8 with checkpointing. The matched no-memory
chronological controls are `v1_current_ep.json` and `v1_sw4_ep.json`.
Prepared wrappers are `train/slurm/v2_r2r.slurm` and `v3_r2r.slurm`; launch from this
branch's root. They request eight CPUs, save under the groups checkpoint folder,
and upload only a completed full-training export to private `anhdao69` repositories.
No new Slurm jobs were submitted; validation used the existing interactive allocation.

The following sections describe the independent v0/v1 IID baselines retained on main.

Plain supervised fine-tuning of public Qwen3.5-4B. The visual backbone is frozen;
the vision merger and language model are trained. v0 uses up to eight uniformly
selected preceding observations plus current. v1 uses the four immediately
preceding observations plus current. There is no recurrent memory or geometry
module in these training paths.

The local R2R dataset has **631,244 states in 10,819 complete episodes**. Every
frame/action was checked against original R2R metadata; rebuilt v0 annotations
match every R2R record in the existing JanusVLN mixed-data JSON. v0 and v1 have
identical targets and instructions. See [the review](implementations/review_v0_v3.md)
and [v2 implementation plan](implementations/v2_memory_only_implementation.md).

## Prepared launchers

**No Slurm jobs were submitted.** The scripts are prepared for a later full run.

- `train/v0_uniform8.sh` with `configs/datasets/newton_r2r_v0.json`
- `train/v1_sw4.sh` with `configs/datasets/newton_r2r_v1.json`
- Cluster wrappers: `train/slurm/v0_r2r.slurm`, `train/slurm/v1_r2r.slurm`

The cluster wrappers request four H10080GB GPUs, 8 CPUs, one image-loading
worker/GPU, and single-threaded CPU math. Global batch is 64; v0 uses batch2 per
GPU and accumulation8; v1 uses batch4 and accumulation4. ZeRO-1, bf16, FA2, fused AdamW, LR1e-6 language / 1e-5
merger, weight decay0.01, one epoch and seed42 are used. v0 batch4/GPU without
activation checkpointing ran out of H100 memory in testing.

Weights go under `/groups/yshang/an221229/checkpoints/StageVLN-v2/`. Slurm runs
get unique job-ID directories, a pinned public checkpoint revision, a source
snapshot, package manifest, and data/prompt protocols. Checkpoint resume must
be explicit; an existing checkpoint directory is not resumed automatically.

After successful training, each Slurm wrapper uploads the final export to its
private `anhdao69/StageVLN-v0-r2r-uniform8` or `anhdao69/StageVLN-v1-r2r-sw4`
repository. Intermediate optimizer checkpoints are excluded. The token is read
from `~/.cache/stagevln/hf_token`, mode0600, never from committed source.
No upload has been performed. The upload helper can retry independently after
a successful training run using `--folder` and `--repo`.

The generic launcher requires an explicitly selected environment (`ENV_DIR`),
an active virtualenv, or this repository's `.venv`. The cluster wrapper selects
the tested shared environment explicitly. Its versions are Python3.12.13,
PyTorch2.10.0+cu129, Transformers5.3.0, Accelerate1.13.0, DeepSpeed0.16.4,
FlashAttention2.8.3. Always set `PYTHONPATH=$PWD/src` for direct Python commands
because that shared environment also contains another project's editable install.

## Correctness changes

The collator rejects overlength examples and validates image tokens per sample.
The loss averages target tokens within each action state and normalizes using
the actual global accumulation-window state count. It does not weight longer
action names more heavily. Only real target-prediction positions receive
vocabulary logits; dense/selected gradients are tested for unequal prompt lengths.
The saved tokenizer and prompt protocol preserve the exact non-thinking prefix.
Custom normalization weights and biases are excluded from decay.

Accelerate pads the epoch tail with four repeated examples: 631,248 exposures,
with a final update of16 rather than64. This is recorded, and normalized by the
actual count. The recurrent episode trainer instead consumes each of the 631,244
states exactly once, with a final update of12 labels and no padded observations.

## Rebuild and check

```bash
/usr/bin/python3 scripts/data/prepare_r2r.py
export PYTHONPATH="$PWD/src"
export ENV_DIR=/home/an221229/code/SpatialForcing-VLN/.venv
"$ENV_DIR/bin/python" -m unittest discover -s tests -v
"$ENV_DIR/bin/python" scripts/train/check_prompt.py
```

`prepare_r2r.py` writes the two annotation files, complete episode JSONL,
smoke samples, and hashes under `/groups/yshang/an221229/data/StageVLN-v2/`.
It checks contiguous frame positions and every action against original ground
truth before publishing its outputs.

The review distinguishes unit tests and bounded GPU checks from unrun full
training, upload, and simulator evaluation. No navigation-quality claim is made.
