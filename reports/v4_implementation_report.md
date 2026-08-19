# SpatialForcing v4 Implementation Report

Date: 2026-08-19
Repository: `/work2/11528/anhdao69/stampede3/code/SpatialStack`
Implementation branch: `v4`
Starting point: `main` at commit `867a359`
Experiment: SpatialForcing v0 + GeoVR-style current-frame depth supervision

## 1. Executive summary

SpatialForcing v4 is implemented and verified on the `v4` branch. It preserves
the v0 navigation and feature-alignment path and adds one training-only dense
depth objective:

```text
L_total = L_nav + 0.3 * L_SF + 0.05 * L_depth
L_depth = L_reg + L_grad
```

The implementation uses the same frozen VGGT current-frame computation for the
existing layer-23 SF target and the new VGGT depth pseudo-label. Qwen is run
once, VGGT's aggregator is run once, layer 24 is captured once and reused, and
the depth head consumes selected Qwen hidden states `[7, 16, 24, 32]` in
explicit shallow-to-deep order.

The actual GeoVR source and supplied paper were used as the reference. The
ported decoder retains GeoVR's projection sizes, DPT refinement topology,
normalized sine/cosine UV embeddings, pixel shuffle, final exponential
positivity transform, 98% residual filter, and four strided gradient scales.
Camera, scale, confidence, heading, point, track, causal-memory, and
inference-time geometry additions were intentionally excluded.

Verification completed:

- 31/31 repository tests passed;
- Python compilation and all three shell launchers passed syntax checks;
- 100 real R2R examples passed forward/alignment/depth validation;
- a four-H100, 100-optimizer-step R2R training smoke completed;
- depth loss reached both the student depth head and Qwen;
- VGGT and its depth head had zero gradients on every logged step;
- the saved depth head changed numerically and restored correctly;
- DeepSpeed checkpoint resume was tested and fixed;
- the final checkpoint loaded through the stock Qwen class with no teacher or
  auxiliary inference modules and produced a finite real navigation loss.

The final selected depth settings remain:

```text
depth_loss_weight = 0.05
depth_head_lr      = 1e-5
```

The R2R server data contains some broken trajectory references. A pre-existing
fallback bug that repeatedly selected the same adjacent broken trajectory was
fixed so training can recover using distinct random records. This changes only
the behavior of unreadable samples.

RxR could not be validated on this allocation because its actual media is not
installed. The full R2R+RxR launcher is delivered, but its dataset preflight
will correctly stop until a valid combined config/media root is supplied.

## 2. Scientific scope and invariants

The controlled question is:

> Does explicit VGGT pseudo-depth provide useful geometric supervision beyond
> SpatialForcing v0's VGGT feature-alignment objective?

The retained v0 settings are:

| Setting | Value |
|---|---:|
| `spatial_forcing_enabled` | `true` |
| `sf_loss_weight` | `0.3` |
| `sf_student_layer` | `24` |
| `sf_teacher_layer` | `23` |
| `sf_projector_hidden_dim` | `4096` |
| `sf_use_vggt_pe` | `true` |
| `sf_multiframe_teacher` | `false` |
| `use_geometry_encoder` | `false` |
| `use_geometry_fusion` | `false` |

The only scientific addition is GeoVR-style explicit depth supervision from
the same current observation. The implementation enforces:

1. one Qwen forward;
2. one VGGT aggregator forward per sample;
3. current-frame-only VGGT input;
4. no teacher gradient;
5. no geometry feature injection into Qwen;
6. exact current-image token selection at all four student layers;
7. exact layer-24 tensor reuse for SF and depth;
8. matching student/teacher dense shapes;
9. positive finite student depth;
10. training-only auxiliary modules, with stock Qwen inference.

## 3. References audited before implementation

The pre-implementation audit is in
`reports/v4_geovr_depth_code_audit.md`. The key references were:

- `papers/GeoVR.pdf`, Sections 3.2, 3.4, 3.7, 4.1, and Table 6;
- `GeoVR-MLLM/models/dense_head.py::DenseHead`;
- `GeoVR-MLLM/models/qwen3vl_geo.py::extract_hidden_states`;
- `::get_corresponding_qwen_indexes`;
- `::get_geo_features_vggt`;
- `::filter_by_quantile`;
- `::_compute_depth_loss`;
- `GeoVR-MLLM/training/trainer.py::create_optimizer`;
- `GeoVR-MLLM/training/train.py::safe_save_model_for_hf_trainer`; and
- `GeoVR-MLLM/scripts/train.sh`.

GeoVR's released DenseHead indexes absolute positions from a full hidden-state
tuple. SpatialStack deliberately retains only four selected tensors, so copying
that absolute indexing would be wrong. The v4 head instead accepts exactly four
tensors in `[H7, H16, H24, H32]` order and applies projection levels 0–3 by
relative list position. A unit test rejects a full 33-entry/absolute-index-style
input.

## 4. Final architecture

```text
instruction + history + current RGB
                  |
             Qwen3.5-VL
                  |
       one forward; capture current-image tokens
          H7, H16, H24, H32
              |          |
        H24 reused        +--> GeoVRDepthHead --> positive dense D_student
              |                                      |
      SF projector                                  L_depth
              |                                      ^
              v                                      |
        cosine SF loss                    cropped VGGT pseudo-depth
                                                     ^
current Qwen RGB crop --> white bottom/right pad --> frozen VGGT aggregator
                                                     |
                               one shared token list +--> layer-23 SF feature
                                                     +--> frozen VGGT depth head
```

### 4.1 Student depth head

`GeoVRDepthHead` has 33,777,969 trainable parameters and uses:

```text
input dimension       = 2560
student merged patch  = 16 * 2 = 32 pixels
target patch basis    = 14 * 2 = 28 pixels
input hidden states   = [7, 16, 24, 32]
projection channels  = [256, 512, 1024, 1024]
resize factors        = [4, 2, 1, 0.5]
fusion channels       = 256
UV encoding scale     = 0.1
UV frequency base     = 100
final shuffle factor  = 28 / 4 = 7
output activation     = exp
```

Each input tensor has shape `[B, Hs*Ws, 2560]`. It is LayerNormed, reshaped to
`[B, 2560, Hs, Ws]`, projected, position encoded, and resized. Four DPT levels
are fused deepest-to-shallowest with residual convolution units and bilinear
interpolation (`align_corners=True`), matching GeoVR.

The final order is intentionally exact:

```text
1x1 projection -> pixel_shuffle(7) -> BHWC permutation -> exp
```

Qwen's 384x512 field of view is not divisible by 7 or 28. The implementation
therefore decodes the smallest covering canvas using `ceil(target/7)` and crops
the positive result at the bottom/right to the exact target. This is the
smallest necessary adaptation to GeoVR's released fixed, divisible resolutions;
it preserves GeoVR's layers and activation order while satisfying the required
exact 384x512 loss correspondence.

### 4.2 Hidden-state indexing semantics

For the installed Qwen3.5 implementation:

- hidden-state entry 0 is the input embedding state;
- entries 1–31 are the outputs of decoder blocks 0–30;
- entry 32 is the last decoder output after the language model's final RMSNorm.

The hooks therefore map:

| Requested hidden entry | Producing module |
|---:|---|
| 7 | decoder block 6 |
| 16 | decoder block 15 |
| 24 | decoder block 23 |
| 32 | final language-model RMSNorm |

This preserves the existing v0 layer-24 convention. The capture set is a union,
so layer 24 has one hook and one tensor object. An identity assertion confirms
the tensor used by the SF branch is the same object passed to the depth branch.

### 4.3 Single-pass frozen teacher

`VGGTEncoder.encode_features_and_depth` calls:

```text
aggregated_tokens_list, patch_start_idx = vggt.aggregator(images)
```

once. The same `aggregated_tokens_list` is then passed to:

- the existing layer extractor for the SF feature; and
- `vggt.depth_head(...)` for depth and confidence.

The wrapper is inside `torch.no_grad`, every teacher parameter has
`requires_grad=False`, and `train()` always returns the teacher to evaluation
mode. Unit call-counting and all 100 real-data/training steps reported exactly
one aggregator call. Both `vggt_has_gradient` and
`vggt_depth_has_gradient` were zero for every smoke step.

VGGT confidence is logged diagnostically but never weights the student loss.

## 5. Image alignment and the 16-versus-14 patch issue

The old v0 preprocessing independently reloaded and anisotropically resized the
teacher image to a VGGT grid. That is unsuitable for pixelwise loss because it
does not guarantee the same crop/FOV as Qwen.

With depth enabled, `prepare_image_inputs` now reuses the exact tensor that
produced Qwen's visual input. For the verified R2R examples:

```text
Qwen source crop:        384 x 512
Qwen patch grid:          24 x 32 (patch 16)
Qwen merged token grid:   12 x 16 (merge 2)
VGGT required patch:           14
```

VGGT refuses inputs whose dimensions are not multiples of 14. The teacher-only
wrapper pads white pixels only on the bottom and right:

```text
384 x 512 -> 392 x 518
padding      8 x 6
VGGT grid   28 x 37 = 1,036 tokens
```

VGGT dense output is cropped back to 384x512 before loss. No original Qwen
pixel is resized, discarded, or shifted. SF positional encoding uses the padded
teacher aspect/grid because those features span the padded VGGT canvas, then
the 2-D feature grid is bilinearly resized to Qwen's 12x16 token grid.

This alignment path is gated by `depth_supervision_enabled`. When depth is
disabled, the original v0 teacher preprocessing and feature-only VGGT
configuration remain intact, with no depth-head load or extra capture hooks.

## 6. Exact depth loss

Let valid teacher pixels be:

```text
M = isfinite(D_teacher) and D_teacher > 0
E = D_student - D_teacher
```

The regression term is GeoVR's per-pixel absolute residual after its exact 98%
filter:

```text
r = abs(E[M])
L_reg = mean(filter_by_quantile(r, valid_range=0.98))
```

The filter behavior is preserved:

1. tensors with at most 1,000 elements bypass filtering;
2. thresholds use a detached copy clamped at 100;
3. tensors over 10,000,000 elements sample 1,000,000 threshold values;
4. `k = ceil(0.98 * (N - 1)) + 1` and `torch.kthvalue` are used;
5. the mask is applied to the original gradient-carrying residual;
6. fewer than 1,000 survivors fall back to the full tensor clamped at 100.

For steps `s in [1, 2, 4, 8]`:

```text
E_s  = E[..., ::s, ::s]
Gx_s = clamp(abs(E_s[..., :, 1:] - E_s[..., :, :-1]), max=100)
Gy_s = clamp(abs(E_s[..., 1:, :] - E_s[..., :-1, :]), max=100)
```

Only adjacent pairs whose two teacher pixels are valid contribute. A scale with
fewer than ten valid pixels is skipped. Each valid scale contributes the sum of
its x and y means, and valid scales are averaged:

```text
L_grad  = mean_s(mean(Gx_s[valid_x]) + mean(Gy_s[valid_y]))
L_depth = L_reg + L_grad
```

All-invalid teacher maps return a graph-connected zero. Student NaN/Inf or
non-positive values fail immediately. Teacher invalid fractions, valid
fraction, distributions, raw/weighted losses, and MAE are logged.

No SILog, median/scale alignment, teacher-confidence weighting, or extra
inter-term coefficient was added.

## 7. Configuration and optimizer

New model arguments:

| Argument | v4 value |
|---|---:|
| `sf_multiframe_teacher` | `false` |
| `depth_supervision_enabled` | `true` |
| `depth_loss_weight` | `0.05` |
| `depth_student_layers` | `7 16 24 32` |
| `depth_loss_type` | `geo_depth` |
| `depth_gradient_scales` | `1 2 4 8` |
| `depth_outlier_keep_ratio` | `0.98` |
| `depth_use_teacher_confidence` | `false` |

New training argument:

```text
depth_head_lr = 1e-5
```

The optimizer constructs disjoint decay/no-decay groups for:

```text
base Qwen parameters      -> learning_rate (1e-6)
SF projector/merger       -> mm_projector_lr (1e-5)
student depth head        -> depth_head_lr (1e-5)
```

It asserts that every trainable parameter occurs exactly once and that all
trainable depth-head parameters occur in the depth group. GeoVR uses `1e-4` for
its geometry head, but v4 deliberately uses the plan's safer `1e-5` because the
VLN backbone is trained at `1e-6`.

The 100-step log confirmed the configured depth LR at every step and the
expected cosine-scheduled live LR.

## 8. Checkpointing, resume, and inference

### 8.1 Checkpoint contents

The frozen teacher is reproducible from its source checkpoint and is excluded
from `state_dict`. The 100-step safetensors file contains:

```text
total tensors             788
student depth tensors      58
SF projector tensors        6
VGGT teacher tensors         0
```

The depth head therefore persists, while VGGT does not inflate inference
checkpoints.

### 8.2 DeepSpeed resume fix and proof

The initial one-step checkpoint revealed that DeepSpeed strict restore expected
the intentionally omitted `spatial_teacher.*` keys. `load_state_dict` now
allows only those teacher omissions; missing/unexpected Qwen, SF, or depth keys
still raise a strict-load error.

Resume was rerun from `checkpoint-1` to step 3 and completed. Comparing the
step-1 checkpoint with the resumed final safetensors showed:

```text
depth tensors present       = 58
depth tensors changed       = 57
mean absolute tensor delta  = 3.11469e-7
maximum absolute delta      = 1.52588e-5
VGGT tensors saved          = 0
```

The only unchanged depth tensor was compatible with a zero/fixed bias under the
very short schedule. Live output-gradient hooks and the tensor delta together
prove backward reachability and actual optimizer updates.

### 8.3 Teacher-free inference

The final checkpoint was loaded explicitly with stock
`transformers.Qwen3_5ForConditionalGeneration`, not the training subclass. The
training-only SF/depth keys were reported as expected unexpected task-head keys
and ignored. No `spatial_teacher`, `spatial_projector`, `student_depth_head`,
`geometry_encoder`, or `geometry_merger` module existed on the loaded model.

A real R2R forward produced:

```json
{
  "model_class": "Qwen3_5ForConditionalGeneration",
  "forbidden_modules_initialized": [],
  "finite_navigation_loss": 0.22441375255584717,
  "vggt_initialized": false,
  "projector_initialized": false,
  "geometry_fusion_initialized": false
}
```

Thus inference remains stock Qwen and does not execute VGGT/depth code.

## 9. File-by-file implementation inventory

### New v4 files

- `src/qwen_vl/model/depth_supervision.py`: GeoVR-style depth decoder, UV
  embeddings, robust residual filter, exact L1+gradient objective, finite-depth
  statistics, validation, and graph-connected invalid-map handling.
- `scripts/train/train_spatial_forcing_vln_v4.sh`: controlled v4 wrapper with
  depth enabled, v0 PE enabled, lambda 0.05, and head LR 1e-5; supports bounded
  smoke overrides.
- `scripts/train/train_spatial_forcing_vln_v4_full.sh`: one-epoch full
  R2R+RxR wrapper matched to `train_spatial_forcing_vln_full.sh`.
- `scripts/validation/validate_depth_supervision.py`: real-record invariant,
  distribution, and qualitative panel validator.
- `tests/test_depth_supervision.py`: head ordering/shape/positivity/gradient,
  loss/filter edges, UV determinism, one-aggregator execution, VGGT padding,
  hidden-state mapping, preprocessing reuse, optimizer grouping, checkpoint
  round trip, and strict teacher-omission restore tests.
- `tests/test_data_retry.py`: verifies distinct fallback indices can escape a
  corrupt/missing trajectory cluster.

### Modified files

- `src/qwen_vl/model/spatial_forcing.py`: depth module construction, four-layer
  capture, layer-24 reuse, single shared teacher path, loss integration,
  metrics, live gradient checks, teacher-free state dict, strict resume rule.
- `src/qwen_vl/model/geometry_encoders/vggt_encoder.py`: optional pretrained
  depth head, refactored v0 feature extraction, shared feature+depth execution,
  Qwen-FOV padding/cropping metadata, CPU-safe autocast handling.
- `src/qwen_vl/data/utils.py`: depth-gated exact reuse of the Qwen RGB crop;
  byte-for-byte original v0 preprocessing otherwise.
- `src/qwen_vl/data/data_qwen.py`: depth flag propagation for only the current
  frame and robust distinct-record fallback for missing local media.
- `src/qwen_vl/train/argument.py`: v4 CLI/dataclass fields and head LR.
- `src/qwen_vl/train/train_qwen.py`: controlled-v4 validation, config plumbing,
  frozen teacher/trainable depth setup, and data flag plumbing.
- `src/qwen_vl/train/trainer.py`: disjoint optimizer groups, depth metrics,
  current/configured depth LR, Qwen/head gradient proof, teacher zero-gradient
  proof.
- `scripts/train/train_spatial_forcing_vln.sh`: depth CLI/env plumbing and
  append-safe resume logs; depth defaults off for v0.
- `scripts/validation/validate_spatial_forcing_inference.py`: config-based
  server data selection, arbitrary sample index, and explicit depth-module
  absence check.

### Restored VGGT source required by `enable_depth=True`

The branch's model loader referenced these modules, but main contained only
stale bytecode for them. The corresponding source was restored from the local
`spatialstack-working` branch:

```text
src/qwen_vl/model/vggt/heads/camera_head.py
src/qwen_vl/model/vggt/heads/dpt_head.py
src/qwen_vl/model/vggt/heads/head_act.py
src/qwen_vl/model/vggt/heads/track_head.py
src/qwen_vl/model/vggt/heads/utils.py
src/qwen_vl/model/vggt/heads/track_modules/__init__.py
src/qwen_vl/model/vggt/heads/track_modules/base_track_predictor.py
src/qwen_vl/model/vggt/heads/track_modules/blocks.py
src/qwen_vl/model/vggt/heads/track_modules/modules.py
src/qwen_vl/model/vggt/heads/track_modules/utils.py
src/qwen_vl/model/vggt/utils/geometry.py
src/qwen_vl/model/vggt/utils/helper.py
src/qwen_vl/model/vggt/utils/load_fn.py
src/qwen_vl/model/vggt/utils/pose_enc.py
src/qwen_vl/model/vggt/utils/rotation.py
```

Only the pretrained VGGT DPT depth head is enabled by v4. Camera, point, and
track heads remain disabled.

## 10. Automated tests

Commands:

```bash
.venv/bin/python -m py_compile \
  src/qwen_vl/model/depth_supervision.py \
  src/qwen_vl/model/geometry_encoders/vggt_encoder.py \
  src/qwen_vl/model/spatial_forcing.py \
  src/qwen_vl/train/argument.py \
  src/qwen_vl/train/train_qwen.py \
  src/qwen_vl/train/trainer.py \
  src/qwen_vl/data/utils.py \
  src/qwen_vl/data/data_qwen.py \
  scripts/validation/validate_depth_supervision.py

bash -n scripts/train/train_spatial_forcing_vln.sh \
  scripts/train/train_spatial_forcing_vln_v4.sh \
  scripts/train/train_spatial_forcing_vln_v4_full.sh

.venv/bin/pytest tests -q
```

Final result:

```text
31 passed, 23 warnings in 9.19s
```

Warnings are upstream deprecation notices from Hugging Face, TorchScript,
Pydantic, and DeepSpeed. `git diff --check` passed.

A bare repository-root `pytest` also discovers the supplied external
`Spatial-Forcing` reference tree and fails on its optional JAX/Flax
dependencies. The project-owned suite is `pytest tests`; none of the 31 project
tests failed.

## 11. Real-data validation

The full validation report is
`reports/v0_depth_pretraining_validation.md`. On 100 R2R examples:

| Check | Result |
|---|---:|
| samples | 100 |
| exact current Qwen tokens | 192/192 each |
| exact depth size | 384x512 each |
| aggregator calls/sample | 1.0 |
| teacher valid fraction | 1.0 |
| teacher NaN fraction | 0.0 |
| teacher Inf fraction | 0.0 |
| teacher non-positive fraction | 0.0 |
| teacher min across set | 0.129138 |
| teacher max across set | 4.210157 |
| teacher mean | 0.858483 |
| initial raw depth loss | 0.450335 |
| initial weighted depth loss | 0.022517 |

All recorded scalars were finite. Four initial and four trained qualitative
panels are retained under `reports/v4_validation_r2r_100/` and
`reports/v4_validation_r2r_trained_step100/`.

## 12. Four-H100 smoke training

### 12.1 Environment

```text
Slurm job       3419545
node            c563-004
GPUs            4 x NVIDIA H100 (95,830 MiB each)
PyTorch         2.10.0+cu129
CUDA runtime    12.9
Triton          3.7.1
precision       bfloat16 + TF32
DeepSpeed       ZeRO stage 2
microbatch      1/GPU
grad accumulation 1
global batch    4
dataset         janusvln_r2r
smoke subset    first 1,000 annotations with robust fallback
optimizer steps 100
```

The successful output is:

```text
output/v4_depth_smoke_r2r_100steps_20260819_retry/
```

Training runtime was 262.215 seconds at 0.381 optimizer steps/second and 1.525
samples/second.

### 12.2 First-ten versus last-ten metrics

| Metric | First 10 mean | Last 10 mean | Direction |
|---|---:|---:|---|
| total loss | 0.956035 | 0.356534 | down |
| navigation loss | 0.632759 | 0.153737 | down |
| SF loss | 0.997019 | 0.605370 | down |
| SF cosine | 0.002981 | 0.394630 | up |
| raw depth loss | 0.483419 | 0.423712 | down |
| depth regression | 0.374123 | 0.355795 | down |
| depth gradient | 0.109296 | 0.067916 | down |
| depth MAE | 0.388695 | 0.380322 | slightly down/noisy |
| weighted depth | 0.024171 | 0.021186 | down |
| predicted depth mean | 1.008154 | 0.854796 | adapts toward teacher mean |

Per-example pseudo-depth difficulty varies, so the windowed depth metrics are
more meaningful than comparing a single first and last record. The gradient
component decreased clearly. The predicted map did not collapse: over the last
ten records, per-record extrema averaged approximately 0.746–1.028, and the
step-100 map itself spanned 0.741–1.018.

Across all 100 steps:

```text
all logged numbers finite                 yes
depth valid fraction                      1.0 every step
projector gradient verified               1.0 every step
SF student gradient verified              1.0 every step
depth-head output gradient verified       1.0 every step
Qwen depth-path gradient verified         1.0 every step
VGGT has gradient                         0.0 every step
VGGT depth head has gradient              0.0 every step
VGGT aggregator calls/sample              1.0 every step
```

### 12.3 Lambda decision

`lambda_depth=0.05` is retained. On the independent 100-example startup
validation, weighted depth averaged 0.0225 versus weighted SF 0.2930. During
training, weighted depth averaged 0.0242 in the first ten and 0.0212 in the last
ten. It therefore contributes a real gradient without dominating navigation or
SF. No architecture change or lambda sweep is justified before the short pilot.

### 12.4 Dataset defect encountered and corrected

The server's R2R annotation includes references to missing media directories,
including trajectory IDs 1 and 10. The old dataset retry logic attempted the
same bad index three times and then the same `i+1` index three times; adjacent
records often belong to the same missing trajectory, causing distributed
training to terminate.

The fallback now samples up to ten unique nonzero offsets, yielding distinct
indices and excluding the original record. A test covers this behavior. The
successful 100-step run encountered missing records repeatedly and recovered
without changing any valid sample's content or preprocessing.

## 13. Launchers and exact commands

### 13.1 Bounded R2R smoke

```bash
MODEL_PATH=/scratch/11528/anhdao69/model_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
TEACHER_MODEL_PATH=/scratch/11528/anhdao69/model_cache/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9 \
CACHE_DIR=/scratch/11528/anhdao69/model_cache \
DATASET_CONFIG=$PWD/configs/datasets/janusvln_r2r.json \
DATASET_USE=janusvln_r2r \
OUTPUT_DIR=$PWD/output/v4_depth_smoke_r2r \
NPROC_PER_NODE=4 \
MAX_STEPS=100 \
MAX_SAMPLES=1000 \
GRADIENT_ACCUMULATION_STEPS=1 \
DATALOADER_NUM_WORKERS=0 \
LOGGING_STEPS=1 \
WARMUP_STEPS=1 \
SAVE_STRATEGY=no \
bash scripts/train/train_spatial_forcing_vln_v4.sh
```

### 13.2 Full controlled R2R+RxR run

The delivered script is:

```text
scripts/train/train_spatial_forcing_vln_v4_full.sh
```

It mirrors `train_spatial_forcing_vln_full.sh`: one full epoch, all samples,
v0 positional encoding, 3% warmup, step checkpoints every 1,000 steps, and no
`MAX_STEPS` cap. The only scientific additions are the v4 depth flags.

Run it after installing a valid combined annotation/media config:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
TEACHER_MODEL_PATH=/path/to/VGGT-1B \
CACHE_DIR=/path/to/cache \
DATASET_CONFIG=/path/to/valid_janusvln_r2r_rxr.json \
OUTPUT_DIR=$PWD/output/spatial_forcing_vln_v4_depth_r2r_rxr_epoch1 \
NPROC_PER_NODE=4 \
bash scripts/train/train_spatial_forcing_vln_v4_full.sh
```

The base launcher preflights the annotation file and media directory before
allocating model replicas. On this server, the checked-in combined config will
currently fail that preflight because `/mnt/data/vmo-ai-task/...` is absent.
The local 5.2 GB `train_r2r_rxr_parquet.json` alone is insufficient because its
referenced parquet media files are not installed.

## 14. Acceptance checklist

| Requirement | Status | Evidence |
|---|---|---|
| implement on `v4`, not main | pass | active branch `v4` |
| preserve v0 when depth off | pass | gated construction/data/hooks; tests |
| inspect paper and supplied code | pass | audit report and cited functions |
| exact four selected student layers | pass | mapping test and runtime metrics |
| layer-24 reuse | pass | identity assertion; metric 1.0 |
| one Qwen forward | pass | one parent forward path |
| one VGGT aggregator | pass | unit mock + 100-example/100-step metric |
| frozen VGGT | pass | requires-grad checks + zero gradients all steps |
| GeoVR-style dense head | pass | architecture port and tests |
| GeoVR-style robust loss | pass | formula/edge tests |
| exact dense shape | pass | 384x512 on all real examples |
| finite positive student depth | pass | runtime failure guards and tests |
| depth reaches Qwen/head | pass | live gradients every smoke step |
| optimizer group exactness | pass | unique-coverage assertion and unit test |
| checkpoint round trip | pass | unit + real trained reload |
| DeepSpeed resume | pass | checkpoint-1 resumed after strict-load fix |
| teacher-free inference | pass | stock Qwen real forward |
| 100 R2R real-data examples | pass | validation JSON/panels |
| 100 optimizer steps | pass | final trainer state global step 100 |
| RxR real-data examples | blocked by data | parquet media unavailable |
| full launcher | pass | executable + shell syntax + preflight |

## 15. Controlled-experiment status and next step

The implementation and R2R smoke are healthy enough for the plan's short
2k–4k pilot. A long combined run should not start until:

1. a valid RxR media installation/config is supplied;
2. 100 RxR examples pass the same validator;
3. the short pilot is compared with the same-step v0 control; and
4. an intermediate checkpoint is evaluated on R2R val-unseen.

The result table therefore remains scientifically honest:

| Method | SF | Depth | Heading | SR | SPL | OS | NE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT |  |  |  | TBD | TBD | TBD | TBD |
| v0 | yes |  |  | 48.50 | 41.86 | 61.83 | 6.33 |
| v0 + Depth (v4) | yes | yes |  | TBD | TBD | TBD | TBD |
| corrected v2 | yes |  | yes | TBD | TBD | TBD | TBD |

No navigation success-rate claim is made from a 100-step training smoke. The
smoke establishes implementation correctness, stability, gradient routing,
checkpoint behavior, and a defensible loss scale; the pilot/evaluation must
answer whether explicit depth improves navigation.
