# Technical Plan — v2 Action-Derived History-to-Current Heading Supervision

## 1. Status and Purpose

This plan extends the verified v0 Spatial Forcing implementation with one
training-only, trajectory-aware auxiliary objective.

The v2 objective is to make Qwen's layer-24 representations encode the camera
heading change from every selected historical observation to the current
observation.

The complete training objective is:

\[
L_{\text{total}}
=
L_{\text{nav}}
+ 0.3L_{\text{SF-current}}
+ 0.1L_{\text{heading}}
\]

This is **action-derived auxiliary supervision**, not teacher-model
distillation: the heading targets come from executed Habitat actions, while
VGGT remains the teacher only for the unchanged v0 Spatial Forcing objective.

This document specifies an implementable first controlled experiment. It does
not assume that lower heading loss guarantees better navigation. Navigation
improvement must be established by comparing v2 against an otherwise identical
v0 baseline using simulator metrics.

---

## 2. Goals and Non-Goals

### Goals

1. Preserve the existing v0 navigation and Spatial Forcing paths.
2. Reuse the layer-24 hidden tensor from the existing Qwen forward pass.
3. Extract a representation for every selected history/current frame.
4. Predict the action-derived yaw change from each history frame to the current
   frame.
5. Backpropagate the heading loss into the heading modules and trainable Qwen
   modules.
6. Add negligible compute relative to Qwen3.5-4B and VGGT-1B.
7. Add no model, input, cache, or compute requirement at inference.

### Non-goals

- Do not change the navigation prompt or action labels.
- Do not change which frames Qwen receives.
- Do not run Qwen or VGGT a second time.
- Do not send historical frames to VGGT.
- Do not modify the v0 current-frame token selection or SF loss.
- Do not add geometry fusion.
- Do not infer translation, distance, or a complete relative pose in v2.
- Do not claim improved SR/SPL/NE from training-loss behavior alone.

---

## 3. Preserve v0 Exactly

The existing v0 behavior remains:

```text
Qwen input:
    instruction
    + up to 8 selected history frames
    + 1 current frame

VGGT input:
    current frame only

Spatial Forcing:
    Qwen layer 24 current-frame tokens
    -> existing SF projector
    -> cosine alignment with VGGT layer 23
```

Keep the following values explicit in both the v0 baseline and v2 launchers:

```text
sf_loss_weight             = 0.3
sf_student_layer           = 24
sf_teacher_layer           = 23
sf_use_vggt_pe             = true
use_geometry_encoder       = false
use_geometry_fusion        = false
tune_mm_vision             = false
tune_mm_llm                = true
tune_mm_mlp                = true
Qwen/merger learning rate  = 1e-6
SF projector learning rate = 1e-5
```

`sf_use_vggt_pe` must be passed explicitly for both comparison runs. The base
v0 launcher and full launcher currently have different defaults, so relying on
an implicit default would not produce a controlled experiment.

The new feature must be guarded by:

```text
heading_supervision_enabled = false  # default
```

When the flag is false:

- do not create heading modules;
- do not require heading data fields;
- do not calculate heading targets;
- do not change v0's total loss or metrics;
- produce the same v0 result for identical inputs and weights.

---

## 4. Exact Heading Convention

### 4.1 Observation/action timing

For the verified JanusVLN R2R data, a file named:

```text
step_k_ACTION.png
```

contains observation \(I_k\) captured **before** `ACTION` is executed. The
action in the filename moves the simulator from observation \(I_k\) toward
observation \(I_{k+1}\).

Therefore, the heading change from historical observation \(I_i\) to current
observation \(I_t\) uses exactly:

```text
actions i, i+1, ..., t-1
```

The action encoded by the current frame, \(a_t\), must be excluded. This is the
half-open interval `[i, t)`.

### 4.2 Sign convention

Define the supervised quantity as:

> current camera heading minus historical camera heading

or:

\[
\Delta\theta_{i\rightarrow t}
=
\operatorname{wrap}(\theta_t-\theta_i)
\]

Use:

```text
TURN_LEFT    = +15 degrees
TURN_RIGHT   = -15 degrees
MOVE_FORWARD =   0 degrees
STOP         =   0 degrees
```

Thus:

\[
\Delta\theta_{i\rightarrow t}
=
\operatorname{wrap}
\left(
15^\circ
\sum_{k=i}^{t-1}
s(a_k)
\right)
\]

where \(s(\text{TURN_LEFT})=+1\),
\(s(\text{TURN_RIGHT})=-1\), and the other actions map to zero.

Do not describe this target as “the history heading relative to current,” which
can imply the opposite sign. Use `history_to_current_heading` or
`current_minus_history_heading` consistently in code and documentation.

### 4.3 Wrapping and representation

Wrap radians into:

\[
[-\pi,\pi)
\]

using:

```python
wrapped = torch.remainder(angle_rad + torch.pi, 2 * torch.pi) - torch.pi
```

Represent the target as:

\[
y_{i,t}
=
[\sin(\Delta\theta_{i\rightarrow t}),
  \cos(\Delta\theta_{i\rightarrow t})]
\]

The target order is always `[sin, cos]`.

### 4.4 Example

For history step 10 and current step 15:

```text
step 10: MOVE_FORWARD
step 11: TURN_LEFT
step 12: MOVE_FORWARD
step 13: TURN_LEFT
step 14: TURN_RIGHT
step 15: current action -- excluded
```

Then:

```text
net turns = +1
heading   = +15 degrees
target    = [sin(15 degrees), cos(15 degrees)]
```

---

## 5. Dataset Compatibility and Validation

### 5.1 Initial supported dataset

The first implementation is required to support and validate:

```text
/scratch/11528/anhdao69/data/JanusVLN_data/train_r2r.json
```

The available R2R data has already been checked at plan-review time:

```text
episodes:                         10,819
records/frames:                  631,244
history-to-current pairs:      4,660,468
invalid filenames:                    0
episodes with missing steps:          0
STOP actions:                     10,819
```

RxR must not be declared supported until its configured annotation/media paths
are available and the same validator passes. If a dataset uses a different
turn angle, filename convention, observation/action timing, or non-discrete
camera rotation, it requires a dataset-specific target builder.

### 5.2 Filename parsing

Use one anchored parser. For the current JanusVLN media, accept:

```regex
^step_(\d+)_(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\.(png|jpg)$
```

Do not recover actions with a loose underscore split.

For every record, validate:

1. All selected frame paths are relative to the configured dataset root after
   the existing path normalization.
2. All frames belong to the same episode directory and dataset split.
3. Step indices are strictly increasing.
4. The final selected path is the record's current frame.
5. The current step is greater than every history step.
6. The episode action index contains one and only one file for every required
   step in `[history_step, current_step)`.
7. Every parsed action is supported.
8. The number of targets equals `frame_count - 1`.
9. Every generated sine/cosine target is finite and has norm approximately 1.

Fail loudly on invalid data. Do not silently mask malformed records in the
first controlled experiment.

---

## 6. Target Preprocessing and Cache

Directory traversal must never happen in `__getitem__`.

### 6.1 Required preprocessing script

Add a separate read-only-to-dataset preprocessing command, for example:

```text
scripts/preprocess_heading_targets.py
```

It must:

1. Load the selected annotation and preserve annotation record order.
2. Build each episode's complete `step -> action` index once.
3. Validate the episode and all selected history/current pairs.
4. Compute wrapped heading bins/targets using `[i, t)`.
5. Write the cache atomically through a temporary file followed by rename.
6. Print dataset counts, action counts, heading-bin counts, and validation
   failures.

Training processes must not race to create this cache. Production training
should fail with a clear preprocessing command when the cache is missing or
stale. A bounded smoke validator may build an in-memory index once, but it must
not scan directories per sample.

### 6.2 Compact cache format

Avoid a Python dictionary keyed by full image-path tuples. For 631,244 records,
that representation would add large pickle and string overhead.

Use fixed-width tensors indexed by annotation record ordinal:

```python
{
    "schema_version": 1,
    "target_convention": "current_minus_history",
    "target_order": "sin_cos",
    "turn_angle_deg": 15.0,
    "max_history_frames": 8,
    "annotation_fingerprint": "...",
    "relative_heading_bins": Int8Tensor[num_records, 8],
    "relative_heading_mask": BoolTensor[num_records, 8],
    "frame_count": UInt8Tensor[num_records],
    "record_key_hash": UInt64Tensor[num_records],
}
```

`relative_heading_bins` stores the wrapped number of 15-degree turns in
`[-12, 11]`; `-12` represents 180 degrees under the `[-pi, pi)` convention.
The dataset converts valid bins to float32 `[sin, cos]` tensors at load time.

The fingerprint must cover at least:

- normalized annotation path and content identity;
- ordered record IDs/current paths;
- target convention and schema version;
- maximum history count;
- turn angle.

On load, verify the fingerprint, record count, record-key hashes, frame counts,
and masks. Refuse stale or mismatched caches.

### 6.3 Dataset output

For a record containing `N` frames, return:

```python
{
    # existing v0/Qwen fields
    ...,
    "frame_count": LongTensor[],                 # N
    "current_frame_index": LongTensor[],         # N - 1
    "ordered_frame_paths": tuple[str, ...],      # existing metadata

    # new v2 fields before collation
    "frame_image_token_masks": BoolTensor[N, L],
    "frame_image_token_counts": LongTensor[N],
    "relative_heading_targets": FloatTensor[N - 1, 2],
    "relative_heading_mask": BoolTensor[N - 1],
}
```

The mask is mandatory, even though every unpadded R2R target should be valid.
It is required for collation and future dataset extensions.

Do not cache Qwen hidden states, pooled features, or projected features. They
must be computed live so heading gradients reach Qwen.

---

## 7. Per-Frame Qwen Token Masks

### 7.1 Preserve the v0 current-frame mask

The existing `build_current_image_token_mask` remains the source of truth for
v0 Spatial Forcing. When v2 is enabled, additionally construct masks for all
frames and assert:

```python
torch.equal(frame_image_token_masks[-1], current_image_token_mask)
```

Do not replace or relax the existing current-frame assertions.

### 7.2 New all-frame mask builder

Add a helper that receives:

```text
input_ids
expected post-merge token count for every frame
image token ID
vision-start token ID
vision-end token ID
```

It must:

1. Parse every explicit
   `<|vision_start|> ... <|vision_end|>` image span in sequence order.
2. Require exactly one span per ordered frame path/grid.
3. Select only `<|image_pad|>` tokens inside each corresponding span.
4. Verify each mask count against that frame's Qwen post-merge grid.
5. Reject overlapping spans or masks.
6. Return `[frame_count, sequence_length]` boolean masks.

Do not assume:

- equal token counts between frames;
- a fixed number of image tokens;
- that image tokens form a suffix;
- that the last fixed number of tokens is the current image.

### 7.3 Collation

The collator must produce:

```text
frame_image_token_masks   [B, F_max, L_max]
frame_valid_mask          [B, F_max]
frame_image_token_counts  [B, F_max]
relative_heading_targets  [B, H_max, 2]
relative_heading_mask     [B, H_max]
frame_count               [B]
current_frame_index       [B]
```

where `F_max <= 9` and `H_max = F_max - 1` for the batch.

Sequence padding/truncation must match `input_ids`. After applying
`model_max_length`, verify that every valid frame mask still contains its
expected token count. Reject a batch if any history or current image span was
truncated.

For every sample, assert:

```text
current_frame_index == frame_count - 1
frame_valid_mask.sum() == frame_count
relative_heading_mask.sum() == frame_count - 1
all frame masks are disjoint
last valid frame mask == current_image_token_mask
```

---

## 8. Reuse One Qwen Forward

The current v0 forward hook captures the complete layer-24 hidden tensor:

```text
student_hidden: [B, L, D_student]
```

Reuse this tensor for both objectives:

```text
v0 SF branch:
    select current tokens only

v2 heading branch:
    select and pool tokens for every frame
```

Register only one layer hook and run Qwen exactly once.

Derive dimensions from the loaded model configuration:

```python
student_dim = int(config.text_config.hidden_size)
```

Do not hard-code 2560, even though Qwen3.5-4B currently uses that dimension.

### 8.1 Masked mean pooling

For each valid frame \(f\):

\[
z_f
=
\frac{
\sum_{j=1}^{L}m_{f,j}H_j
}{
\max(\sum_j m_{f,j},1)
}
\]

giving:

```text
z: [B, F_max, D_student]
```

Implement masked pooling without materializing a
`[B, F_max, L, D_student]` tensor. An einsum or equivalent masked reduction is
sufficient because `F_max <= 9`.

Mean pooling is intentionally the lightweight v2 baseline. It may discard
spatial correspondence information that is useful for rotation estimation. Do
not introduce attention pooling or token matching in this first controlled
run; consider those only as follow-up ablations if mean pooling fails.

---

## 9. Heading Modules

Create the heading modules only when `heading_supervision_enabled=True`.

### 9.1 Shared frame projector

Use one shared projector for history and current frames:

```text
Linear(student_dim -> heading_projection_dim)
LayerNorm(heading_projection_dim)
```

Initial value:

```text
heading_projection_dim = 512
```

For each frame:

\[
e_f=P_h(z_f)
\]

### 9.2 Shared relation head

For each valid history frame \(i\), pair it with current frame \(t\):

\[
q_{i,t}
=
[e_i, e_t, e_t-e_i, e_i\odot e_t]
\]

The relation head is:

```text
Linear(4 * heading_projection_dim -> heading_projection_dim)
GELU
Linear(heading_projection_dim -> 2)
```

Its output order is:

```text
[predicted_sin, predicted_cos]
```

Use the same projector and relation head for every frame and pair. Initialize
both with the model's existing initialization routine.

---

## 10. Heading Loss and Metrics

### 10.1 Stable normalization

Calculate predictions and loss in float32:

```python
normalized_prediction = torch.nn.functional.normalize(
    prediction.float(), dim=-1, eps=1e-6
)
target = target.float()
```

Assert that valid targets are finite and unit norm within tolerance.

### 10.2 Per-pair and per-sample loss

For a valid pair:

\[
\ell_{i,t}
=
1-\tilde y_{i,t}^{T}y_{i,t}
\]

First average valid pairs within each sample. Then average only samples that
contain at least one history frame:

\[
L_{\text{heading}}
=
\frac{
\sum_b \mathbf{1}[n_b>0]
\left(
\frac{1}{n_b}\sum_i m_{b,i}\ell_{b,i}
\right)
}{
\max(\sum_b \mathbf{1}[n_b>0],1)
}
\]

This prevents records with eight history frames from receiving eight times the
sample weight of records with one history frame.

### 10.3 Zero-history/DDP behavior

A current-only sample has zero heading pairs. It must:

- produce `heading_loss = 0`;
- never divide by zero;
- produce finite metrics;
- keep the heading computation graph connected so distributed training does
  not see unused heading parameters.

The recommended implementation evaluates the shared heading modules on padded
slots for every batch and masks invalid losses. Clamp denominators to at least
one. If an entire batch has no valid pair, the masked sum must remain a
differentiable zero connected to both heading modules and `student_hidden`.

Do not claim that a nonzero student gradient in the combined objective proves
the heading gradient route: navigation and SF also reach Qwen. Prove the route
in a focused validator by backpropagating `heading_loss` alone.

### 10.4 Angular metric

For monitoring:

```python
pred_angle = torch.atan2(predicted_sin, predicted_cos)
true_angle = torch.atan2(target_sin, target_cos)
wrapped_error = torch.atan2(
    torch.sin(pred_angle - true_angle),
    torch.cos(pred_angle - true_angle),
).abs()
angular_error_deg = torch.rad2deg(wrapped_error)
```

Do not use an unwrapped `abs(pred_angle - true_angle)`.

### 10.5 Required logs

Log valid-pair averages for:

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

Retain all existing v0 logs.

The training log may report that heading-module parameters received gradients
under the combined objective. The heading-only validator is the authoritative
proof that heading loss reaches Qwen.

---

## 11. Final Objective and Gradient Routing

When heading supervision is enabled:

```python
total_loss = (
    navigation_loss
    + sf_loss_weight * spatial_forcing_loss
    + heading_loss_weight * heading_loss
)
```

Initial values:

```text
sf_loss_weight      = 0.3
heading_loss_weight = 0.1
```

Expected gradient routes:

```text
navigation loss
    -> Qwen and navigation head

Spatial Forcing loss
    -> existing SF projector
    -> Qwen through current-frame layer-24 representation

heading loss
    -> heading relation head
    -> heading frame projector
    -> history/current layer-24 representations
    -> trainable Qwen language layers up to the captured layer
    -> trainable multimodal merger

VGGT
    frozen
    no gradients
```

The Qwen vision tower remains frozen. The heading loss can propagate through
its outputs but must not make its parameters trainable.

---

## 12. Optimizer and Configuration Plumbing

### 12.1 Exact argument names

Add the following model arguments:

```python
heading_supervision_enabled: bool = False
heading_loss_weight: float = 0.1
heading_projection_dim: int = 512
heading_turn_angle_deg: float = 15.0
```

Add the following data argument:

```python
heading_cache_path: Optional[str] = None
```

Add the following training argument:

```python
heading_head_lr: Optional[float] = None
```

Do not add multiple aliases such as both `heading_distillation_enabled` and
`heading_supervision_enabled`. Use the names above consistently.

### 12.2 Training entry point

Thread the model arguments into `config` before `from_pretrained`. After parsing,
copy `heading_supervision_enabled` and `heading_turn_angle_deg` into `data_args`
because model and data arguments are separate dataclasses.

For this v2 experiment, require:

```text
spatial_forcing_enabled = true
heading_supervision_enabled = true
use_geometry_encoder = false
use_geometry_fusion = false
data_flatten = false
```

Explicitly mark both heading modules trainable in `set_model` and assert that
they exist when the feature is enabled.

### 12.3 Optimizer groups

The current custom optimizer recognizes only the merger and
`spatial_projector` as projector parameters. Extend it to recognize:

```text
heading_projector
heading_relation_head
```

Use:

```text
heading_head_lr if provided
otherwise mm_projector_lr
```

Initial controlled-run value:

```text
heading_head_lr = 1e-5
```

Do not train the randomly initialized heading modules at the Qwen base LR of
`1e-6` unless an explicit ablation requests it.

### 12.4 Launcher

Add explicit environment/CLI controls:

```text
HEADING_SUPERVISION_ENABLED=True
HEADING_LOSS_WEIGHT=0.1
HEADING_PROJECTION_DIM=512
HEADING_TURN_ANGLE_DEG=15.0
HEADING_HEAD_LR=1e-5
HEADING_CACHE_PATH=/path/to/cache.pt
```

The YAML plan/config file is documentation only in the current repository. The
training launcher constructs CLI arguments directly, so updating YAML alone is
not sufficient.

---

## 13. Checkpoint and Inference Behavior

### 13.1 Training checkpoints

Heading projector and relation-head tensors must be included in training
checkpoints so training can resume exactly.

Update missing-key patterns so loading the original base Qwen checkpoint into
the v2 wrapper does not produce misleading warnings for newly initialized:

```text
heading_projector.*
heading_relation_head.*
```

Add a resume test that:

1. performs an optimizer step;
2. saves a checkpoint;
3. reloads the v2 wrapper;
4. verifies heading weights are identical;
5. resumes with finite losses.

### 13.2 Teacher-free stock-Qwen inference

Inference continues to load the checkpoint with stock:

```text
Qwen3_5ForConditionalGeneration
```

Stock Qwen may report the saved training-only tensors as ignored keys. It must
not instantiate or execute:

```text
VGGT
SF projector
heading projector
heading relation head
geometry fusion
heading targets/cache
```

Inference input and compute remain:

```text
instruction + RGB history/current -> Qwen -> action
```

Therefore v2 has zero additional inference-time module cost.

---

## 14. Required Unit Tests

### 14.1 Action and target construction

1. Exact parser accepts all supported actions and rejects malformed names.
2. `TURN_LEFT -> +15 degrees`.
3. `TURN_RIGHT -> -15 degrees`.
4. `MOVE_FORWARD` and `STOP` contribute zero.
5. History step 10 to current step 20 includes actions 10 through 19.
6. The action in step 20 is explicitly excluded.
7. Skipped selected frames still use every intermediate action.
8. `375 degrees -> 15 degrees`.
9. `-375 degrees -> -15 degrees`.
10. `180 degrees -> -180 degrees` under `[-pi, pi)`.
11. Every `[sin, cos]` target has unit norm.
12. Cross-episode paths, duplicate steps, gaps, reverse order, and unknown
    actions raise clear errors.

### 14.2 Cache

1. Cache round-trip preserves bins, masks, counts, and record order.
2. Annotation fingerprint mismatch is rejected.
3. Turn-angle or convention mismatch is rejected.
4. Record-key mismatch is rejected.
5. Training does not traverse episode directories in `__getitem__`.

### 14.3 Per-frame token extraction

Test 1-, 2-, 4-, and 9-frame records, including different token counts per
frame:

- exactly one mask per frame;
- each count matches its own grid;
- masks are disjoint;
- the last all-frame mask equals the unchanged v0 current mask;
- prompt, instruction, action, padding, and other frames are excluded;
- truncating any valid image span is rejected.

### 14.4 Loss behavior

1. One current frame only: finite differentiable zero heading loss.
2. Two frames: one valid pair.
3. Nine frames: eight valid pairs.
4. Mixed-history batch: correct masks and per-sample averaging.
5. Identical prediction/target: zero cosine loss.
6. Opposite prediction/target: cosine loss two.
7. Angular error wraps correctly across `-pi/pi`.
8. Zero-length prediction is numerically safe.
9. No NaN or Inf under bf16 model execution with float32 loss math.

### 14.5 Gradient isolation

Backpropagate **heading loss alone** and verify:

- nonzero relation-head gradients for a valid pair;
- nonzero heading-projector gradients;
- nonzero captured Qwen/student gradients;
- no VGGT gradients;
- frozen Qwen vision-tower parameters remain without gradients;
- the zero-history case does not trigger an unused-parameter failure.

### 14.6 v0 equivalence and checkpointing

1. With heading disabled, identical inputs/weights give the same v0 navigation,
   SF, and total losses.
2. Original v0 data does not require a heading cache.
3. A saved v2 wrapper restores heading weights when resuming.
4. The stock-Qwen inference validator loads the checkpoint without creating
   training-only modules.

---

## 15. Real-Data and Distributed Validation

### 15.1 Preprocessing validation

Run the target builder across the complete R2R annotation and require:

```text
record count                     = 631,244
episode count                    = 10,819
missing or duplicate steps       = 0
invalid actions/filenames        = 0
invalid record/frame ordering    = 0
invalid target norms             = 0
target count per record          = frame_count - 1
```

Print the observed counts rather than hard-coding them into the parser; the
values above are acceptance expectations for the current data snapshot.

### 15.2 Real forward/backward validator

Use real records containing 1, 2, 4, and 9 frames. Require:

- unchanged v0 current Qwen token count;
- one valid pooled feature per frame;
- finite navigation, SF, heading, and total losses;
- correct heading pair counts `0, 1, 3, 8`;
- heading-only gradients reach the heading modules and Qwen;
- VGGT remains frozen and gradient-free.

### 15.3 Distributed smoke test

Run 100 optimizer steps with the same four-H100, bf16, ZeRO-2 configuration as
v0. The smoke test passes when:

```text
no shape, cache, DDP, or unused-parameter errors
no NaN or Inf
v0 SF invariants still pass
heading loss and metrics are finite
valid batches produce nonzero heading-module gradients
zero-history samples are handled safely
checkpoint save/resume succeeds
stock-Qwen inference validation succeeds
```

A decreasing training heading loss is an optimization check, not evidence of
navigation improvement.

---

## 16. Baselines and Scientific Acceptance

### 16.1 Heading baseline

The current R2R target distribution is centered near zero. Across the verified
4,660,468 pairs, an always-zero-degree predictor has approximately:

```text
cosine loss:            0.451
angular MAE:           47.9 degrees
exact zero targets:    16.1 percent
targets within +/-15:  38.0 percent
```

Therefore, v2 must be evaluated on held-out episodes/scenes against at least:

```text
constant 0-degree prediction
best constant heading estimated on the training split
```

The learned head should produce lower held-out cosine loss and angular MAE than
these constant baselines. Training loss alone is insufficient because the head
can learn the dataset prior.

Also report performance by:

- absolute heading bin;
- history-to-current step gap;
- number of selected history frames.

This helps detect a shortcut based mainly on temporal distance or frame order.

### 16.2 Navigation comparison

The primary experiment is:

```text
v0: nav + 0.3 * SF
v2: nav + 0.3 * SF + 0.1 * heading
```

Hold constant:

- model initialization/base checkpoint;
- R2R training records and order;
- frame selection;
- prompt and targets;
- random seed;
- optimizer and schedule;
- Qwen/SF learning rates;
- VGGT positional-encoding setting;
- batch size and gradient accumulation;
- epoch/step count;
- evaluation episodes.

Evaluate both checkpoints in the JanusVLN/Habitat navigation loop and report at
least:

```text
Success Rate (SR)
Success weighted by Path Length (SPL)
Navigation Error (NE)
```

One matched-seed comparison is sufficient for the initial decision. If v2
appears better, repeat with at least three seeds before claiming a robust gain.

### 16.3 Decision rule

Call v2 technically successful only if all implementation/gradient/inference
tests pass.

Call the heading objective useful only if it beats the constant heading
baselines on held-out data.

Call v2 a navigation improvement only if simulator metrics improve over the
matched v0 baseline. The value `heading_loss_weight=0.1` is an initial setting,
not a guaranteed optimum; tune it only after the controlled first comparison.

---

## 17. Expected Performance Cost

For Qwen3.5-4B with `student_dim=2560` and projection dimension 512, the new
modules contain approximately 2.36 million parameters.

Expected training overhead is small because:

- the layer-24 hidden tensor is already captured by v0;
- no extra Qwen forward is performed;
- no extra VGGT forward is performed;
- at most eight relation pairs exist per sample;
- pooled frame tensors are small;
- target lookup is from a compact cache.

Training throughput may decrease slightly; v2 is not expected to make training
faster. Inference throughput and memory must remain unchanged because stock
Qwen ignores all training-only modules.

---

## 18. Implementation Order

Implement and verify in this order:

1. Add strict filename/action parsing and heading-target utilities.
2. Add the offline compact-cache builder and cache validation.
3. Add per-frame Qwen token-mask construction and focused unit tests.
4. Add dataset outputs and padded collation with truncation checks.
5. Add heading projector, relation head, masked pooling, and loss utilities.
6. Reuse the existing single layer-24 hook for SF and heading.
7. Add total-loss integration behind the default-false flag.
8. Add explicit trainability, optimizer groups, CLI arguments, and launcher
   controls.
9. Add distributed metric aggregation and gradient diagnostics.
10. Add checkpoint-resume and stock-Qwen inference tests.
11. Run complete R2R preprocessing validation.
12. Run real 1/2/4/9-frame forward/backward validation.
13. Run the 100-step four-H100 smoke test.
14. Train matched v0 and v2 checkpoints and evaluate SR/SPL/NE.

Do not begin the full training comparison until steps 1-13 pass.

---

## 19. Expected Final System

### Training

```text
instruction + history/current RGB
                |
                v
              Qwen
                |
        +-------+-------------------+
        |                           |
        v                           v
navigation prediction       layer-24 frame tokens
        |                           |
        v                    +------+------+
navigation loss              |             |
                              v             v
                      current tokens    mean pool every frame
                              |             |
                              v             v
                      existing SF loss  heading relation loss
                              ^
                              |
                     frozen VGGT(current)
```

```text
total = navigation + 0.3 * current-frame SF + 0.1 * heading
```

### Inference

```text
instruction + history/current RGB -> stock Qwen -> action
```

No VGGT, SF projector, heading modules, action cache, geometry fusion, or
spatial cache is initialized or required at inference.
