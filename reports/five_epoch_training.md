# Five-epoch v2 training and recovery

The `re_mem` job uses v2 recurrent memory, R2R only, four H100 80GB GPUs,
global labeled-state batch 64, reader microbatch 4, K=8, native sharded
Adam optimizer state, BF16 autocast, and FP32 trainable weights, gradients
and optimizer moments. It requests eight CPUs and 192GB host memory, with
one data-loader worker per rank and one CPU compute thread per rank.

## Epoch accounting

Each epoch consumes all 10,819 episodes and 631,244 labeled observations
once. Its 9,864 updates include a final batch of 12 labels. The final partial
batch is not combined with the following epoch. Five epochs therefore mean
49,320 optimizer updates and 3,156,220 labeled observations. The cosine LR
schedule spans all 49,320 updates with one initial warmup update. Neither
Adam moments nor the LR schedule restart at epoch boundaries.

Episode order uses `seed + epoch - 1`. Whole-episode rank assignments and
chronological frame order are preserved. Every stream must be exhausted,
including any unlabeled terminal observations, before advancing the epoch;
the trainer must have cleared its episode memory carry at that point.

## Saving and retention

The run directory is
`/groups/yshang/an221229/checkpoints/StageVLN-v2/re_mem_JOBID`.

* Every 500 global optimizer updates, save `step-epoch-E-STEP` and retain
  the five newest complete periodic checkpoints of the current epoch.
  At the previously measured throughput, this is approximately 18 minutes
  between recovery points, excluding I/O and contention.
* At every epoch boundary, save a permanent full `epoch-E` checkpoint and
  an inference export under `epoch-E/export`. All five are retained.
* Only after the epoch checkpoint and export finish, remove that epoch's
  periodic checkpoints. The permanent epoch checkpoint is the recovery
  point until the next periodic save in the following epoch.
* Checkpoints include model, each rank's optimizer shard, LR scheduler,
  RNG states, epoch/permutation/cursors, and recurrent memory carry. A
  completion manifest is written atomically after all ranks save.
* Interrupted partial saves have no completion manifest and must not be
  selected for recovery. A retry can replace their temporary payloads.

Full checkpoints are about 49GiB and exports about 17GiB with this model.
Budget approximately 600GiB for retained artifacts and an in-progress save.
The filesystem has ample free capacity; group quota reports no explicit
limit. The account's user quota report is already above its listed limit,
so that report alone does not establish future write availability.

## Hugging Face publication

A separate uploader watches atomic `EPOCH_COMPLETE` markers and pushes
each immutable export while GPU training continues. The destinations are
private repositories `anhdao69/StageVLN-v2_mem64_r0-r2r-epoch1` through
`anhdao69/StageVLN-v2_mem64_r0-r2r-epoch5`. Actual authenticated test commits
to all five destinations succeeded before submission. Credentials remain
in the user's mode-0600 cache file, outside source and training artifacts.

Optimizer shards stay on the cluster. A successful upload writes an
`HF_UPLOAD_COMPLETE` marker. Uploads retry failures up to ten times;
resuming the uploader skips confirmed uploads. Failure never deletes
local exports, and the launcher reports unsuccessful publication through
its exit code. Smoke exports cannot be published by this uploader.

## Resume

Use the same run directory, its saved source snapshot, and four GPUs.
Choose the latest checkpoint with a complete `manifest.json`; compare its
global `step`, then `epoch` if steps are equal. Do not choose an incomplete
directory or roll back past an already completed later epoch.

```bash
export RUN_DIR=/groups/yshang/an221229/checkpoints/StageVLN-v2/re_mem_JOBID
export RESUME_CHECKPOINT="$RUN_DIR/step-epoch-E-STEP"
sbatch --export=ALL train/slurm/re_mem.slurm
```

The runner strictly checks source/config identity and restores optimizer,
LR, RNG, and stream state. An exhausted saved stream finalizes its existing
epoch idempotently, then advances once. Mid-epoch recovery continues at
the next unconsumed observation. The existing source snapshot is reused
on resume. A completed epoch checkpoint can also be selected directly.

To retry only uploads, run the snapshot's `scripts/recurrent/upload_epochs.py`
with `--run "$RUN_DIR" --prefix anhdao69/StageVLN-v2_mem64_r0-r2r`.

## Timing and validation scope

Previous representative v2 measurement was approximately 5.92 compute
hours per epoch. Allow roughly 30–40 hours for five epochs, plus checkpoint
and upload overhead; the Slurm limit is 48 hours. Queue time is additional.
These are estimates, not a completed full-training benchmark.

New regression coverage checks five separate epoch tails, deterministic
shuffle and cursor restoration, exact optimizer/model/RNG continuation at
an epoch boundary and within epoch two, continuous cosine LR reaching zero
after five epochs, retention exclusions, and upload completion/idempotence.

Validation on 2026-09-18: the complete suite passed **54 tests** in 27.04s.
The real runner was also exercised with tiny CPU components, real checkpoint
files, all five epoch exports, and a mid-epoch interruption: resumed final
weights matched uninterrupted training bit for bit. The existing two-rank
optimizer-shard resume test also passed in this suite.

A four-H100, full 4B-model smoke run completed five epochs of one 39-state
episode (five optimizer updates), with loss
`1.029716, 1.029716, 0.540762, 0.300164, 0.276418`, finite gradients and
nonzero writer gradients. LR reached zero at step five, as configured for
that bounded smoke dataset. Peak GPU allocation was 64.35GiB. This smoke
disabled saving; production save/export orchestration was covered by the
CPU runner test and full-model checkpoint/export had been validated in the
earlier recurrent implementation report. The smoke is intentionally excluded
from full-epoch throughput estimation because all its batches are partial.

The submitted job uses a frozen archive of the committed source so later
worktree edits cannot change the queued training. Slurm's dry run accepted
the four-H100, eight-CPU, 192GB, 48-hour request.
