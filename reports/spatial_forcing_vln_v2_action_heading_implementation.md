# SpatialForcing-VLN v2 action-heading implementation report

- Date: 2026-08-10
- Branch: `v2`
- Plan implemented: `plans/v2_action_heading.md`
Status: complete implementation, complete R2R target preprocessing, real-data
adapter validation, local test suite, real H100 forward/backward validation,
100-step four-GPU training smoke, exact ZeRO-2 resume, and stock-Qwen inference
validation. The final successful artifacts are under
`/scratch/11528/anhdao69/spatialstack_runs/sf_v2_r2r_smoke_3388496_retry2`.

## 1. Resulting method

v2 preserves the v0 model/data flow and adds one training-only action-derived
heading loss:

```text
navigation branch:
instruction + selected history/current RGB -> Qwen -> navigation CE

unchanged v0 branch:
Qwen layer-24 current tokens -> SF projector
current RGB -> frozen VGGT layer 23 -> 2-D resize
projected Qwen / frozen VGGT -> cosine Spatial Forcing loss

new v2 branch:
same captured Qwen layer-24 tensor
    -> explicit tokens for every selected frame
    -> masked mean per frame
    -> shared frame projector
    -> pair every history frame with current
    -> shared relation head
    -> action-derived [sin(yaw), cos(yaw)] cosine loss
```

The implemented training objective is:

```text
total_loss = navigation_loss + 0.3 * spatial_forcing_loss + 0.1 * heading_loss
```

The heading path performs no additional Qwen forward and no additional VGGT
forward. Historical images are still never sent to VGGT. Geometry encoding and
fusion remain disabled.

## 2. Exact heading convention

The implementation uses `current_minus_history` consistently. A filename
`step_k_ACTION.png` is observation `I_k` before `ACTION`. The target from
history step `i` to current step `t` therefore accumulates exactly actions
`[i, t)` and explicitly excludes the action in the current-frame filename.

Action signs are:

```text
TURN_LEFT     +1 turn = +15 degrees
TURN_RIGHT    -1 turn = -15 degrees
MOVE_FORWARD   0
STOP           0
```

Integer turn bins are wrapped to `[-12, 11]`; bin `-12` is -180 degrees under
the required `[-pi, pi)` convention. Dataset targets are generated in float32
and always ordered `[sin, cos]`.

The real-data check confirms the sign/timing convention. R2R record 1 pairs
step 0 to current step 1. Step 0 executed `TURN_RIGHT`, and its target is:

```text
[-0.25881898, 0.96592587] = [sin(-15 deg), cos(-15 deg)]
```

## 3. Offline cache and full R2R validation

### Strict media/action parsing

`src/qwen_vl/data/heading_targets.py` uses the anchored expression:

```regex
^step_(\d+)_(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\.(png|jpg)$
```

The offline builder:

1. preserves annotation ordinal order;
2. normalizes every selected path relative to the configured data root;
3. validates same-episode membership, strictly increasing selected steps, and
   final-frame/current-record identity;
4. scans each referenced episode directory once;
5. rejects unknown filenames, duplicate step files, nonzero starts, and gaps;
6. checks selected-frame filename actions against the complete episode index;
7. accumulates every intermediate action in `[history_step, current_step)`;
8. checks target count, finiteness, and unit norm;
9. writes through a same-directory temporary file and atomic rename.

No directory traversal occurs in dataset `__getitem__`.

### Fixed-width format

The 16-MiB R2R cache contains:

```text
schema_version:            1
target_convention:         current_minus_history
target_order:              sin_cos
turn_angle_deg:            15.0
max_history_frames:        8
relative_heading_bins:     Int8[631244, 8]
relative_heading_mask:     Bool[631244, 8]
frame_count:               UInt8[631244]
record_key_hash:           UInt64[631244]
```

The fingerprint covers the resolved annotation path, SHA-256 content identity,
schema/convention/order/turn angle/history limit, and ordered record-key hashes.
Each record-key hash covers the record ID and complete ordered selected-frame
tuple. Cache load verifies metadata, source SHA-256, fingerprint, tensor shapes
and dtypes, record count, masks against `frame_count - 1`, record hashes, frame
counts, finite targets, and unit target norms. A bounded smoke dataset can use a
prefix of the full cache while preserving original annotation ordinals.

### Complete R2R preprocessing result

Command:

```bash
.venv/bin/python scripts/preprocess_heading_targets.py \
  --annotation-path /scratch/11528/anhdao69/data/JanusVLN_data/train_r2r.json \
  --data-root /scratch/11528/anhdao69/data/JanusVLN_data \
  --output /scratch/11528/anhdao69/data/JanusVLN_data/train_r2r_heading_v2.pt \
  --turn-angle-deg 15 \
  --max-history-frames 8
```

Observed result:

```text
records:                         631,244
episodes:                         10,819
history-to-current pairs:      4,660,468
MOVE_FORWARD actions:            404,912
TURN_LEFT actions:               111,177
TURN_RIGHT actions:              104,336
STOP actions:                     10,819
validation failures:                   0
cache size:                       16 MiB
fingerprint: 842ad5af8a605f0a466f3bce4aea9e7fd2866a7f74329054b5ce30b8c55ce6d6
wall time:                        5 min 19.14 s
maximum resident memory:          3.78 GiB
```

The path normalizer was optimized during this validation. It now resolves and
caches the configured dataset root once and performs lexical containment for
each absolute image path, followed by the existing explicit `..` escape check.
This avoids millions of unnecessary shared-filesystem `stat` calls.

## 4. All-frame Qwen masks and collation

The unchanged v0 `build_current_image_token_mask` remains the source of truth
for Spatial Forcing. When v2 is enabled, `build_frame_image_token_masks` parses
every explicit `<|vision_start|> ... <|vision_end|>` span in sequence order,
selects only `<|image_pad|>` inside the span, validates each frame against its
own post-merge grid, rejects nested/unclosed/extra spans, and rejects overlap.
The final all-frame mask must equal the unchanged v0 current mask exactly.

Before collation, a sample contains:

```text
frame_image_token_masks   [N, L]
frame_image_token_counts  [N]
relative_heading_targets  [N-1, 2]
relative_heading_mask     [N-1]
```

The collator produces:

```text
frame_image_token_masks   [B, F_max, L_max]
frame_valid_mask          [B, F_max]
frame_image_token_counts  [B, F_max]
relative_heading_targets  [B, F_max-1, 2]
relative_heading_mask     [B, F_max-1]
frame_count               [B]
current_frame_index       [B]
```

It validates all valid token counts after model-length truncation, disjoint
masks, current index, valid-frame count, target count, final/current mask
identity, finite targets, and target norms. A real mixed 1/2/4/9-frame batch
found and led to correction of a padding bug that was invisible in batch-size-1
checks: shorter per-sample mask sequences are now copied only into their valid
sequence prefix before batch padding.

Real R2R processor validation passed:

| Frames | Pair count | Tokens for every valid frame |
|---:|---:|---:|
| 1 | 0 | 192 |
| 2 | 1 | 192 |
| 4 | 3 | 192 |
| 9 | 8 | 192 |

## 5. Heading model and loss

### One captured Qwen tensor

The same temporary forward hook already used by v0 captures the complete
layer-24 tensor `[B, L, D_student]`. The hook is registered once, Qwen runs
once, both auxiliary branches reuse the result, and the hook is removed in
`finally`.

Dimensions derive from `config.text_config.hidden_size`; 2560 is not hard-coded.
Frame pooling uses:

```python
torch.einsum("bfl,bld->bfd", mask, hidden) / token_count
```

This avoids materializing `[B, F, L, D]`.

### Modules

The frame projector shared by all history/current frames is:

```text
Linear(student_dim -> 512)
LayerNorm(512)
```

Each history embedding `e_i` is paired with the current embedding `e_t` as:

```text
[e_i, e_t, e_t - e_i, e_i * e_t]
```

The relation head shared by all pairs is:

```text
Linear(2048 -> 512)
GELU
Linear(512 -> 2)  # [predicted_sin, predicted_cos]
```

Both modules use the Qwen model initialization routine. At the production
dimensions they add approximately 2.36 million trainable parameters.

### Loss balance and numerical behavior

Predictions are normalized in float32 with epsilon `1e-6`. Valid target vectors
are asserted finite and unit length. Cosine loss is averaged over pairs inside
each sample first, then over samples containing history. Thus an eight-history
record does not receive eight times the sample weight of a one-history record.

Angular monitoring uses wrapped `atan2(sin(error), cos(error))`, so values
across the -pi/pi boundary are correct.

For an all-current-only batch, the model evaluates both heading modules on a
dummy relation slot and multiplies its loss by a false mask. The result is a
finite differentiable zero connected to the relation head, frame projector,
and captured Qwen tensor, avoiding unused-parameter behavior in DDP.

### Metrics

Existing v0 metrics are retained. v2 adds:

```text
heading_loss
heading_cosine_similarity
heading_mean_angular_error_deg
heading_pair_count
heading_samples_with_history
history_frame_count
heading_projector_grad_verified
heading_relation_head_grad_verified
```

Distributed logging weights heading cosine/angular values by valid pair count
and heading loss by samples containing history. Zero-history records do not
bias either average.

## 6. Gradient routing and optimizer groups

The intended routes are encoded as follows:

```text
heading loss -> relation head -> frame projector
             -> layer-24 history/current representation
             -> trainable Qwen decoder/merger

VGGT: frozen, eval, no_grad, current-only
Qwen vision blocks: requires_grad=False
Qwen merger: requires_grad=True
Qwen language/lm_head: requires_grad=True
```

The optimizer now has three disjoint name sets:

1. Qwen base parameters at `learning_rate` (`1e-6`);
2. merger and `spatial_projector` at `mm_projector_lr` (`1e-5`);
3. `heading_projector` and `heading_relation_head` at `heading_head_lr`, falling
   back to `mm_projector_lr` when unspecified (`1e-5` in v2).

Unit tests verify both the dedicated rate and fallback.

## 7. Configuration, launchers, checkpointing, and inference

Added exact arguments:

```text
--heading_supervision_enabled
--heading_loss_weight
--heading_projection_dim
--heading_turn_angle_deg
--heading_cache_path
--heading_head_lr
```

The default flag is false. Disabled v2 creates neither heading module, requires
no cache/data field, and leaves the v0 objective unchanged.

Heading tensors remain in training checkpoints. Missing-key patterns allow
initializing a v2 wrapper from stock Qwen without misleading missing-key noise.
The Slurm smoke recipe saves a complete ZeRO-2 checkpoint at step 100 and starts
a second launcher at max step 101; the existing training entry automatically
detects `checkpoint-100`, restoring model and optimizer state for the extra
finite step.

Stock-Qwen inference continues to use `Qwen3_5ForConditionalGeneration`. The
updated inference validator rejects initialization of:

```text
spatial_teacher
spatial_projector
heading_projector
heading_relation_head
geometry_encoder
geometry_merger
```

The training-only tensors may be reported as ignored by stock Transformers but
are never instantiated or executed. Inference needs no heading cache/actions.

## 8. Files added or changed

Core implementation:

- `src/qwen_vl/data/heading_targets.py`: parser, target construction, compact
  cache, atomic save, fingerprint and load-time validation.
- `src/qwen_vl/data/data_qwen.py`: fast normalization, cache attachment,
  all-frame masks, targets, and padded collation.
- `src/qwen_vl/model/spatial_forcing.py`: frame pooling, heading modules,
  sample-balanced objective, metrics, zero-history graph, and one-hook reuse.
- `src/qwen_vl/train/argument.py`: exact v2 arguments.
- `src/qwen_vl/train/train_qwen.py`: config/data plumbing and trainability.
- `src/qwen_vl/train/trainer.py`: valid-pair distributed metrics, gradient
  diagnostics, and dedicated optimizer groups.

Commands/configuration:

- `scripts/preprocess_heading_targets.py`
- `scripts/train/train_spatial_forcing_vln_v2.sh`
- `scripts/train/train_spatial_forcing_vln_v2_full_r2r.sh`
- `scripts/train/train_spatial_forcing_vln_v2_full_rxr.sh`
- `scripts/slurm/smoke_spatial_forcing_vln_v2_r2r.slurm`
- `scripts/slurm/train_spatial_forcing_vln_v2_full_r2r.slurm`
- `configs/spatial_forcing_vln_v2_r2r.yaml`
- `configs/spatial_forcing_vln_v2_r2r_rxr.yaml`

Validation/tests:

- `scripts/validation/validate_action_heading_data.py`
- `scripts/validation/validate_action_heading_forward.py`
- updated stock-Qwen inference validator
- `tests/test_heading_targets.py`
- `tests/test_heading_supervision.py`
- `tests/test_heading_optimizer.py`
- `pytest.ini`, limiting discovery to this repository's tests rather than
  ignored local reference repositories.

Documentation/launch plumbing:

- updated `README.md`
- base v0 launcher accepts default-false v2 controls and appends logs on resume.

## 9. Completed verification

### Test suite

```text
52 passed in 4.83 s
```

Coverage includes exact parsing, sign/timing, current-action exclusion, skipped
selected frames, +375/-375/180-degree wrapping, unit targets, malformed episode
relations, cache round-trip and mismatch rejection, 1/2/4/9-frame masks,
different per-frame token counts, mixed collation, history truncation, zero/one/
eight-pair behavior, per-sample loss balancing, identical/opposite cosine loss,
pi-boundary angular error, zero predictions, bf16 target
quantization/renormalization and loss math, heading-only student/module
gradients, checkpoint round-trip, optimizer rates, and all v0 tests.

### Syntax/static checks

All modified Python files compile; every shell/Slurm launcher passes `bash -n`;
`git diff --check` is clean.

### Real data

- Complete R2R cache build: passed all 631,244 records and 10,819 episodes.
- Real Qwen processor samples with 1/2/4/9 frames: passed.
- Mixed real-data collation: passed.
- v0 final-mask equality for every checked sample: passed.
- Cache target sign and pair counts: passed.

## 10. Completed H100 Slurm smoke

### Allocation and exact workflow

The user subsequently confirmed that the previous task in interactive Slurm
job `3388496` was finished and explicitly authorized using that allocation.
The smoke ran on node `c562-001` with four NVIDIA H100 GPUs (95,830 MiB
reported per device). Before launch, all four GPUs had zero resident processes
and zero allocated memory.

The successful workflow was:

1. real one-GPU Qwen+VGGT heading-only forward/backward;
2. 100 optimizer steps on four H100s, bf16, ZeRO-2, batch 1/GPU,
   gradient accumulation 8 (effective batch 32);
3. full model/optimizer/scheduler/RNG checkpoint at step 100;
4. reconstruction of frozen VGGT and exact DeepSpeed resume to step 101;
5. final student save and stock `Qwen3_5ForConditionalGeneration` inference.

### Runtime issues found and corrected

The smoke found two environment/checkpoint issues and one diagnostics issue
that CPU tests could not expose:

1. The original Slurm wrapper loaded system CUDA 12.8 before a PyTorch
   `2.10.0+cu129` environment. `libcusparse.so.12` then resolved against an
   incompatible `libnvJitLink`, and torch import failed before model
   allocation. Both v2 Slurm wrappers now purge modules and load GCC only;
   torch correctly reports CUDA 12.9 and all four H100s.
2. DeepSpeed casts all floating batch inputs to bf16. The float32 heading
   targets were already validated at `1e-5` in the collator, but bf16 component
   quantization can change a 15-degree target norm by about `0.00101`. The
   model now checks against the numerical precision of the received dtype and
   renormalizes in float32 before cosine math. Invalid float32 cache targets
   remain subject to the original strict validation.
3. Frozen VGGT is intentionally excluded from checkpoints, but DeepSpeed's
   strict resume loader initially required its keys. The model loader now
   ignores missing `spatial_teacher.*` keys only after reconstructing VGGT from
   `teacher_model_path`; every student, SF, and heading key remains strict.
   The same step-100 checkpoint then resumed successfully.

The first distributed log also showed zero for the two heading diagnostic
flags even though the modules were training. This was instrumentation-specific:
ZeRO can replace parameter gradient accumulators. The checkpoint proves that
all eight heading tensors were present in the dedicated `1e-5` optimizer groups
and had nonzero Adam first/second moments at step 100. The heading path now
mirrors the v0 SF diagnostic by attaching hooks to module outputs as well as
parameters. The strengthened real backward validator reports both hook flags
as true.

### Real heading-only backward result

The final validator used an actual nine-frame R2R record (eight
history-to-current pairs) and reported:

```text
heading pairs:                         8
Qwen/VGGT resized tokens:              192 / 192
heading loss:                          0.914385
heading mean angular error:            84.6693 degrees
heading projector parameter gradient: true
heading relation-head parameter grad:  true
heading projector output hook:         true
heading relation-head output hook:     true
captured Qwen layer-24 heading grad:    true
Qwen language / merger gradients:      true / true
frozen Qwen vision-block gradient:      false
frozen VGGT gradient:                   false
maximum allocated CUDA memory:          24.6552 GiB
```

The absolute loss is not expected to be low for this validator because it uses
a newly initialized heading head. Its purpose is real-shape, real-data,
gradient-boundary, and numerical validation.

### 100-step loss behavior

All 100 optimizer-step records are finite. Endpoint, ten-step-window, and
least-squares trends from `checkpoint-100/trainer_state.json` are:

| Metric | Step 1 | Step 100 | Mean 1-10 | Mean 91-100 | Slope/step |
|---|---:|---:|---:|---:|---:|
| total loss | 2.55219 | 0.39712 | 1.04993 | 0.40885 | -0.004582 |
| navigation loss | 2.16627 | 0.18444 | 0.69250 | 0.18795 | -0.003117 |
| SF loss | 1.02124 | 0.57236 | 0.99520 | 0.57654 | -0.004564 |
| SF cosine | -0.02124 | 0.42764 | 0.00480 | 0.42346 | +0.004564 |
| heading loss | 0.79551 | 0.40975 | 0.59462 | 0.49048 | -0.000981 |
| heading cosine | 0.16903 | 0.57950 | 0.38634 | 0.48468 | +0.000906 |
| heading angular error | 79.3902 | 46.1426 | 61.5159 | 53.7527 | -0.07291 deg |

The ten-step-window changes are `-0.64108` total loss, `-0.50455`
navigation loss, `-0.41866` SF loss, `-0.10414` heading loss, and `-7.7632`
degrees heading error. Qwen and resized-VGGT counts were exactly `192` at every
step; `vggt_has_gradient` was zero at every step. The average number of valid
pairs per microbatch ranged from `6.53125` to `8.0`; samples-with-history ranged
from `0.9375` to `1.0`, exercising sample-balanced mixed history counts. The
run completed in 455.9 seconds at 0.219 optimizer steps/second.

### Checkpoint, resume, and inference

The step-100 checkpoint is 69 GiB and contains four ZeRO optimizer shards,
model state, scheduler, per-rank RNG state, trainer state, tokenizer, and the
portable 10.39-GB safetensors student. The portable model contains 738 keys,
including all eight heading keys and all six SF-projector keys, and zero frozen
teacher keys.

Resume loaded the exact ZeRO state and executed only step 101. Its metrics were
finite (`total=0.4414`, `navigation=0.2197`, `SF=0.5827`,
`heading=0.5004`, 192/192 tokens, no VGGT gradient). Stock inference then loaded
the final checkpoint using unmodified `Qwen3_5ForConditionalGeneration` and
reported:

```text
model class:                         Qwen3_5ForConditionalGeneration
finite navigation loss:              0.2206804
VGGT initialized:                    false
SF projector initialized:            false
heading modules initialized:         false
geometry fusion initialized:         false
```

Stock Qwen correctly reports the training-only SF/heading tensors as unused
unexpected keys; it neither constructs nor executes those modules.

Important artifact logs are:

```text
heading_only_forward_final.log
train_first_100_steps.log
train.log                         # exact step-100 -> 101 resume
stock_qwen_inference.log
checkpoint-100/trainer_state.json
```

## 11. Full training scripts

### Full R2R on the current server

Preprocess once (already completed for the current snapshot), then submit:

```bash
sbatch scripts/slurm/train_spatial_forcing_vln_v2_full_r2r.slurm
```

or from an existing new allocation:

```bash
bash scripts/train/train_spatial_forcing_vln_v2_full_r2r.sh
```

Defaults include all records, one epoch, four processes, batch 1/GPU,
accumulation 8, base LR `1e-6`, both auxiliary head rates `1e-5`, v0 SF weight
`0.3`, heading weight `0.1`, VGGT PE enabled, 3% warmup, step-1000 saves, and
retention of ten periodic checkpoints.

### R2R+RxR on the separate server

The configured remote paths are:

```text
annotation: /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json
media:      /mnt/data/vmo-ai-task/anhdh35/JanusVLN
cache:      /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr_heading_v2.pt
```

First run the cache builder there. This is mandatory because v2 intentionally
does not claim RxR compatibility until the exact remote annotations, filenames,
timing, turn angle, and episode continuity pass:

```bash
python scripts/preprocess_heading_targets.py \
  --annotation-path /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json \
  --data-root /mnt/data/vmo-ai-task/anhdh35/JanusVLN \
  --output /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr_heading_v2.pt

bash scripts/train/train_spatial_forcing_vln_v2_full_rxr.sh
```

The launcher fails before model allocation if that validated cache is absent.

## 12. Scientific scope

The completed preprocessing, tests, and H100 smoke establish target
correctness, data/model plumbing, numerical behavior, end-to-end gradient
isolation, decreasing training objectives, checkpoint resume, and v0 inference
gating. They do not establish navigation improvement or held-out
generalization.

The next scientific experiment must compare the
matched v0 objective `nav + 0.3 SF` with v2 `nav + 0.3 SF + 0.1 heading`, holding
initialization, data order, seed, frame selection, prompt, optimizer, schedule,
VGGT PE, effective batch, and training duration fixed. Heading prediction must
also beat zero-degree and best-constant held-out baselines. Only matched Habitat
SR/SPL/NE can support a navigation-improvement claim.
