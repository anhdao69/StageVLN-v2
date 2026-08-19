# Implementation Plan: SpatialForcing v0 + GeoVR-Style Depth Supervision

## 0. Objective

Implement a clean **SpatialForcing v0 + depth loss** variant for VLN.

The scientific question is intentionally narrow:

> Does explicit dense depth supervision provide useful geometric information beyond the existing v0 VGGT feature-alignment objective?

This experiment should be treated as a **controlled geometry ablation / GeoVR-style depth baseline**, not as a replacement for the current SpatialForcing formulation and not yet combined with heading, causal-memory, camera, or scale losses.

I will provide the **GeoVR source code** together with the current SpatialForcing-VLN code. **Before implementing anything, inspect the GeoVR implementation carefully and use its actual code as the reference for the depth head, feature extraction, depth loss, initialization, optimizer setup, and tests. Do not recreate GeoVR behavior from memory when the provided code can be inspected directly.**

The implementation should be minimal, correct, well-tested, and preserve v0 behavior exactly when depth supervision is disabled.

---

# 1. Baseline That Must Be Preserved

Start from the existing **SpatialForcing v0** implementation.

The original objective is:

\[
\mathcal{L}_{v0}
=
\mathcal{L}_{nav}
+
0.3\mathcal{L}_{SF}.
\]

Current v0 behavior must remain unchanged:

- Qwen receives instruction + navigation history + current RGB frame.
- Frozen VGGT receives the **current frame only**.
- Existing Spatial Forcing aligns the current Qwen visual representation with the current-frame VGGT representation.
- VGGT is a training-only teacher.
- VGGT and all auxiliary heads are absent from inference.
- Existing evaluation code must remain unchanged.

Preserve the current v0 settings, including:

```yaml
spatial_forcing_enabled: true
sf_loss_weight: 0.3
sf_student_layer: 24
sf_teacher_layer: 23
sf_projector_hidden_dim: 4096
sf_use_vggt_pe: true
sf_multiframe_teacher: false

use_geometry_encoder: false
use_geometry_fusion: false
```

Do not change the Qwen architecture, navigation objective, prompts, history sampling, image inputs, inference behavior, or existing Spatial Forcing target.

---

# 2. New Training Objective

Add one new training-only auxiliary objective:

\[
\boxed{
\mathcal{L}_{total}
=
\mathcal{L}_{nav}
+
0.3\mathcal{L}_{SF}
+
\lambda_{depth}\mathcal{L}_{depth}
}
\]

The new depth target comes from the **same frozen current-frame VGGT teacher** already used by v0.

The intended architecture is:

```text
                       Qwen3.5-VL
 instruction + history + current RGB
                           |
          +----------------+----------------+
          |                                 |
  current tokens @ layer 24       current tokens from
          |                       layers {7,16,24,32}
          |                                 |
     SF projector                  GeoVR-style depth head
          |                                 |
       L_SF                         predicted dense depth
          |                                 |
          |                              L_depth
          |                                 ^
          |                                 |
 Current RGB ---> frozen VGGT aggregator ---+
                         |
                  +------+------+
                  |             |
             layer-23 feat   frozen depth head
                  |             |
               SF target    pseudo-depth target
```

Important invariant:

\[
\boxed{\text{one Qwen forward + one VGGT backbone forward}}
\]

Do **not** run Qwen twice.

Do **not** run the VGGT aggregator twice.

---

# 3. Read and Reuse the GeoVR Code

I will have the GeoVR repository/code available locally.

Before implementation:

1. Locate GeoVR's dense/depth head implementation.
2. Locate how GeoVR selects intermediate Qwen hidden states.
3. Locate how it maps VGGT hierarchy to language-model hierarchy.
4. Locate the exact depth loss implementation.
5. Locate any multi-scale gradient loss implementation.
6. Locate output activation / positivity constraints for depth.
7. Locate positional encoding used by the depth decoder.
8. Locate the optimizer parameter group and learning rate for geometry heads.
9. Locate checkpoint save/load behavior.
10. Locate existing GeoVR depth tests or diagnostics.

Write a short code-audit note before modifying the SpatialForcing repository describing:

- GeoVR files/functions used as references.
- Which parts are ported/adapted.
- Which parts are intentionally omitted.
- Any differences required because our backbone is Qwen3.5-VL rather than GeoVR's original Qwen backbone.
- Any differences required because our task is VLN rather than generic video spatial reasoning.

Do not copy the entire GeoVR architecture unnecessarily. Port only the components needed for **depth-only supervision**.

---

# 4. Teacher: Enable VGGT Depth Prediction

**Confirmed prerequisite (not a hypothetical):** on the current `main` branch, `src/qwen_vl/model/vggt/heads/` contains no tracked depth-head source at all — `dpt_head.py`, `camera_head.py`, `track_head.py`, `head_act.py`, `utils.py`, and their geometry/pose_enc/rotation helpers are absent from git history (only stray, untracked `.pyc` cache files remain). `VGGTEncoder.__init__`/`load_model` (`src/qwen_vl/model/geometry_encoders/vggt_encoder.py`) unconditionally construct VGGT with `enable_camera=enable_point=enable_depth=enable_track=False`, and this is not exposed as a knob through `GeometryEncoderConfig`/`create_geometry_encoder` today. Flipping `enable_depth=True` will not work until these files are restored.

Byte-identical copies of these files exist on branch `spatialstack-working` (and `upstream/main`) and should be cherry-picked from there into `main`'s `src/qwen_vl/model/vggt/heads/` as the very first implementation step (see Section 36, Step 1), before any other depth-supervision work begins.

The current v0 VGGT wrapper may instantiate VGGT with depth disabled because v0 only needs latent features.

For this variant, once the depth-head source above is restored and when:

```yaml
depth_supervision_enabled: true
```

enable the **pretrained frozen VGGT depth head**.

Conceptually:

```python
VGGT(
    ...,
    enable_depth=True,
)
```

while preserving all unrelated settings.

When depth supervision is disabled, preserve the exact existing v0 VGGT initialization.

The teacher must remain:

```text
eval mode
requires_grad = False
torch.no_grad()
```

for:

- VGGT aggregator
- VGGT depth head
- all VGGT-related parameters

No depth loss may backpropagate into VGGT.

---

# 5. Use a Single VGGT Forward for Both v0 Features and Depth

This is a critical implementation requirement.

Bad implementation:

```python
sf_feature = teacher.encode(current_image)
depth = teacher.vggt(current_image)["depth"]
```

if both calls independently execute the VGGT aggregator.

**This is not a hypothetical to guard against — it is exactly what the current code seams would produce if wired up naively.** Today, `VGGTEncoder.encode_layers()` (`src/qwen_vl/model/geometry_encoders/vggt_encoder.py`) — what Spatial Forcing already calls — and `VGGT.forward()` (which internally runs `self.depth_head(...)` when `enable_depth=True`) are two **independent** methods that each call `self.vggt.aggregator(...)` separately. Calling `encode_layers()` for SF and separately calling `self.vggt(images)`/`VGGT.forward()` for depth would run the aggregator twice per training step.

**Concrete fix:** extend `VGGTEncoder` with a new method — e.g. `encode_features_and_depth(current_image, layer_indices, feature_layer=23)` — that:

1. Calls `self.vggt.aggregator(images[None])` **exactly once**, producing `aggregated_tokens_list` and `patch_start_idx`.
2. Slices out the requested SF/depth-pyramid layers from that same `aggregated_tokens_list`, the same way `encode_layers()` already does today (reuse its existing per-layer slicing/reshape logic rather than duplicating it).
3. Calls `self.vggt.depth_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)` **directly** on that same `aggregated_tokens_list` — explicitly bypassing `VGGT.forward()` (which would otherwise call `self.aggregator(...)` a second time) — to obtain the dense depth/confidence maps.

Return something similar to:

```python
{
    "sf_features": ...,
    "depth": ...,
    "depth_conf": ...,
}
```

Internally:

```text
current RGB
   |
VGGT aggregator              <- exactly once
   |
   +---- selected layer-23 representation -> existing SF path
   |
   +---- VGGT depth_head(aggregated_tokens_list, ...) -> pseudo-depth
```

Add an invariant/test that counts or otherwise verifies that the aggregator executes once per teacher forward (e.g. a call-count hook/mock on `self.vggt.aggregator.forward` in a unit test).

---

# 6. Teacher Depth Is Current-Frame Only

Do not use the v1 streaming / causal VGGT teacher for this experiment.

The depth target must be:

\[
D_t^{teacher}
=
VGGT_{depth}(I_t)
\]

where \(I_t\) is only the current RGB frame.

Thus:

```text
Qwen input:
instruction + history + current

VGGT input:
current only
```

This isolates:

\[
\text{v0}
\quad\text{vs}\quad
\text{v0 + explicit current-view depth}.
\]

Set and assert:

```yaml
sf_multiframe_teacher: false
```

Depth supervision must not accidentally instantiate or use the v1 streaming KV-cache pathway.

---

# 7. Student Layers for Depth

Follow the **actual GeoVR layer-selection rule**.

GeoVR uses four hierarchical VGGT levels:

\[
[4,11,17,23]
\]

and proportionally maps them onto the LLM depth.

For our 32-layer Qwen3.5-VL model, this gives:

\[
\boxed{
depth\_student\_layers=[7,16,24,32]
}
\]

using the same hidden-state indexing convention as GeoVR.

Therefore configure:

```yaml
depth_student_layers:
  - 7
  - 16
  - 24
  - 32
```

Existing Spatial Forcing remains:

```yaml
sf_student_layer: 24
sf_teacher_layer: 23
```

Layer 24 should be captured once and reused by both:

- existing Spatial Forcing
- one level of the depth pyramid

Do not perform a second Qwen forward to obtain depth features.

---

# 8. Verify Hidden-State Indexing Before Training

Hidden-state indexing is easy to get wrong.

Before implementing the full depth head, write a unit/integration test that proves exactly which decoder output is returned for indices:

```text
7
16
24
32
```

Document whether the hidden-state list contains:

```text
index 0 = input embedding state
index k = output after decoder block k
```

or a different convention.

Use the **GeoVR source code and current Qwen3.5 implementation** as the source of truth.

Do not assume indexing semantics.

The existing v0 layer-24 extraction must remain numerically unchanged.

---

# 9. Extract Only Current-Image Tokens

Qwen processes multiple frames, but the new depth objective predicts depth only for the current observation.

For each selected hidden state \(H^{(l)}\):

1. Use the existing verified current-image token mask/range.
2. Extract only current-frame visual tokens.
3. Verify the token count.
4. Reshape them into their spatial grid.

Expected current token layout in the current pipeline is approximately:

\[
N=192=12\times16.
\]

For every selected layer:

\[
H_t^{(l)}
\in
\mathbb{R}^{192\times D}
\]

and then:

\[
F_t^{(l)}
\in
\mathbb{R}^{D\times12\times16}.
\]

Required invariant:

```python
current_tokens_layer7.shape[0]  == 192
current_tokens_layer16.shape[0] == 192
current_tokens_layer24.shape[0] == 192
current_tokens_layer32.shape[0] == 192
```

Do not silently reshape if the count is inconsistent.

Fail loudly with diagnostic information.

---

# 10. Student Depth Head

Use the provided **GeoVR dense/depth head as the main implementation reference**.

**Confirmed adaptation requirement — do not reuse GeoVR's `DenseHead.forward` unmodified.** GeoVR's own pipeline builds its per-layer feature list very differently from what Section 9 does here, and the two are not drop-in compatible:

- GeoVR's `extract_hidden_states` (`models/qwen3vl_geo.py`) loops `for layer_hidden_states in all_hidden_states:` over the **entire** LLM hidden-state tuple (every layer, e.g. 29/33 entries for a 28/32-layer model — not just the 4 selected ones) and returns a tuple of that same full length.
- That full-length tuple is passed straight into `DenseHead.forward(..., all_hidden_states=<full tuple>)`.
- Inside `DenseHead.forward` (`models/dense_head.py`): `for feature_idx, layer_idx in enumerate(self.intermediate_layer_idx): x = all_hidden_states[layer_idx]` — this is **absolute** indexing (e.g. `all_hidden_states[24]`), and it is only correct because the list being indexed is the full-length one.

Section 9 of this plan, by design, extracts and reshapes **only** the 4 selected layers (`[7,16,24,32]`) — not the full 33-entry stack — to avoid the memory/compute waste of running current-frame token extraction over every Qwen layer for layers we will never use (GeoVR needed the full stack anyway because it also draws camera-token and scale-token features from it; this depth-only port does not). If the ported head keeps GeoVR's original absolute `intermediate_layer_idx`-based indexing while only being given a 4-element list, `all_hidden_states[24]` on a 4-element list is either an out-of-range error or silently wrong — this is a real bug, not a style choice.

**Required fix:** adapt the ported head to consume exactly the small, ordered list Section 9 produces, using **relative/sequential** indexing instead of GeoVR's absolute one:

```text
Qwen hooks
   |
{H7, H16, H24, H32}
   |
current-frame extraction (Section 9)
   |
[F7, F16, F24, F32]              <- ordered list, length 4
   |
adapted depth head
```

```python
def forward(self, selected_features: list[torch.Tensor], ...):
    multi_scale_features = []
    for feature_idx, x in enumerate(selected_features):
        # NOTE: relative index into the 4-element list, not an absolute
        # GeoVR-style `all_hidden_states[layer_idx]` lookup.
        ...
    ...
```

`depth_student_layers=[7,16,24,32]` remains the config that drives *which* Qwen layers get captured in Section 9's hook/mask logic — it does not need to be re-used as an `intermediate_layer_idx` inside the adapted head itself. Add a unit test that passes a synthetic 4-tensor list through the adapted head and asserts it consumes exactly those 4 tensors in order (no absolute indexing, no dependency on the original layer numbers 7/16/24/32 inside the head).

The intended design is a multi-scale DPT-style head rather than a single linear prediction from one Qwen layer.

Conceptually:

```text
Qwen current features:

layer 7  ----\
layer 16 -----\
layer 24 ------> multi-level GeoVR-style DenseHead -> dense depth
layer 32 -----/
```

Follow GeoVR's implementation for:

- LayerNorm
- channel projections
- hierarchical feature dimensions
- multi-scale resizing
- feature-fusion/refinement blocks
- positional encoding
- depth output activation
- final spatial upsampling
- initialization

Expected hierarchy should follow GeoVR as closely as practical, e.g.:

```text
projected channels:
256
512
1024
1024
```

and the GeoVR multi-scale feature hierarchy.

However, do not hard-code output sizes copied from GeoVR if our Qwen current-image grid or VGGT depth resolution differs.

Infer dimensions from the actual current inputs whenever possible.

---

# 11. Depth Positional Encoding

If GeoVR's DenseHead uses normalized UV positional encoding, preserve it unless code inspection shows a reason not to.

The positional encoding should depend on normalized image/grid coordinates rather than absolute dataset-specific pixel sizes.

Test that positional tensors:

- have the correct H×W,
- are on the correct device,
- use the correct dtype,
- are deterministic,
- do not require gradients unless GeoVR explicitly learns them.

---

# 12. Teacher/Student Spatial Alignment Is a Blocking Requirement

Dense depth loss requires spatial correspondence.

This is stricter than the existing latent SF loss.

**A real misalignment has already been confirmed by code inspection — this is not an open question to investigate from scratch.** Traced exactly in `src/qwen_vl/data/utils.py::prepare_image_inputs`, the `model_type == "qwen3.5"` branch (the one Spatial Forcing actually uses for the current frame):

- **Qwen's own `pixel_values`** come from `load_and_preprocess_images` (a VGGT-style resize to width=518, bicubic, aspect-preserving, with a conditional center-crop of height only if it exceeds 518) followed by a further **bottom-right corner crop** down to the nearest `patch_size * merge_size` multiple — discarding up to ~27px of real image content per axis, content Qwen itself never sees.
- **The VGGT teacher's current-frame input** (`geometry_encoder_inputs` / `sf_teacher_pixel_values`) is built by **reloading the raw image file from disk fresh** (`_load_rgb_image(image)`) and resizing it in **one anisotropic step**, with no cropping at all, directly to `(grid_w * 14, grid_h * 14)` pixels — where `grid_w`/`grid_h` are simply read off Qwen's already-computed patch grid.

**Net effect:** the two token grids match in *size* by construction (teacher dimensions are literally derived from Qwen's grid), but they do **not** represent the same field of view — content near the right/bottom edge that Qwen's crop discarded is still present (squeezed in) on the VGGT side, and the two paths apply different width-vs-height scale factors. This is small in absolute pixel terms (~5% of each axis) but real, and nothing in the codebase checks for it today — only token-count/shape invariants are asserted anywhere (never pixel- or content-level alignment). This affects the *existing* v0 SF loss too, not only the new depth loss — dense depth supervision is simply far more sensitive to it than the coarse, bilinear-tolerant SF cosine-similarity loss.

Notably, the sibling `use_geometry_encoder`/geometry-fusion branch a few lines below in the same function (`elif prepare_geometry: ...`) does **not** have this bug — it reuses the already-resized/cropped Qwen tensor (`images[0]`) directly as the geometry-encoder input, with no raw reload. That is the pattern to copy.

**Concrete fix:** in `prepare_image_inputs`'s `qwen3.5` branch, stop reloading the raw image file for the geometry/VGGT input. Instead reuse the same already-resized/cropped Qwen image tensor that produced `pixel_values`, exactly as the `use_geometry_encoder` branch already does. Gate this fix behind `depth_supervision_enabled` (or verify empirically that it does not materially change v0's own SF loss/metrics before applying it unconditionally) — Section 19 requires v0 behavior to remain exactly unchanged when depth supervision is disabled, and this fix touches teacher-input construction shared by both SF and the new depth path.

Before training, still create the validation utility described below — it now serves as **verification that the fix above worked**, not open-ended discovery:

```text
original image
Qwen-transformed current frame
VGGT-transformed current frame (post-fix)
```

and verifies that normalized locations correspond.

Do not train dense depth against misregistered pixels.

---

# 13. Teacher Depth Target Resolution

Do not reduce the VGGT teacher depth target to the 12×16 Qwen token grid.

The purpose is to test explicit dense geometry.

Let VGGT produce its dense pseudo-depth:

\[
D_t^{teacher}
\in
\mathbb{R}^{H_T\times W_T}.
\]

Configure the student DenseHead to predict:

\[
\hat D_t
\in
\mathbb{R}^{H_T\times W_T}.
\]

The final student output and teacher target must have exactly matching spatial dimensions before loss computation.

If GeoVR predicts at a specific target resolution, follow its resizing logic while adapting it to the actual VGGT output resolution in this repository.

---

# 14. Pseudo-Depth Terminology

The teacher target is:

\[
\boxed{\text{VGGT pseudo-depth}}
\]

not Habitat ground-truth depth.

We are not introducing depth sensors or simulator depth annotations.

The pipeline remains:

```text
RGB only
  |
frozen VGGT during training
  |
pseudo-depth supervision
```

At inference:

```text
instruction + RGB -> Qwen -> action
```

No VGGT.

No depth head.

No depth input.

Do not write or log the target as "ground-truth depth."

---

# 15. Depth Loss

**Confirmed exact formula from the released GeoVR code — no longer speculative.** Read directly from `_compute_depth_loss` (`models/qwen3vl_geo.py`), with the default released config (`model.config.depth_loss_type = "l1"`):

The regression term is per-pixel absolute depth error:

\[
e_p = |\hat D_p - D_p^T|.
\]

GeoVR keeps only the lowest 98% of these per-pixel errors (`filter_by_quantile(loss_reg, valid_range=0.98)`, hard-clamped at 100) before averaging — **this quantile filter operates on the regression-error tensor itself, not on the raw teacher-depth values** (see the correction to Section 17 below):

\[
\mathcal{L}_{reg} = \text{Mean}\left[Q_{0.98}\left(|\hat D - D^T|\right)\right].
\]

**Port GeoVR's `filter_by_quantile()` implementation directly rather than reproducing only its conceptual "keep lowest 98%" behavior — a naive `torch.sort(loss)[...]`-style reimplementation will subtly diverge.** The real helper (`models/qwen3vl_geo.py:65-83`) has several specific edge-case behaviors that must be preserved:

1. Skips filtering entirely and returns the raw tensor unchanged if `loss_tensor.numel() <= min_elements` (default `min_elements=1000`).
2. Computes the quantile threshold from a **detached, hard-clamped** copy of the loss (`loss_tensor.detach().clamp(max=hard_max)`, `hard_max=100.0`) — not from the original (gradient-carrying) tensor.
3. If that copy has more than 10,000,000 elements, subsamples down to 1,000,000 via `torch.randperm` before computing the threshold (a performance guard for very large batches).
4. Estimates the threshold via `torch.kthvalue` at `k = ceil(valid_range * (numel - 1)) + 1`, then clamps the resulting threshold itself to `hard_max`.
5. Applies that threshold as a mask (`loss_tensor <= quantile_thresh`) to the **original, non-detached** loss tensor (so the returned values still carry gradients).
6. If fewer than `min_elements` survive the mask, falls back to `loss_tensor.clamp(max=hard_max)` (the full, unfiltered-but-clamped tensor) instead of the tiny filtered subset.

Reproduce this exact function (signature, defaults, and all six behaviors above), not just its 98th-percentile intent.

The gradient term operates on the pixel-wise difference \(E = \hat D - D^T\) (not on \(\hat D\) and \(D^T\)'s gradients separately, though these are algebraically equivalent for a discrete finite-difference gradient), computed at 4 **strided-subsampling** scales (`diff[..., ::step, ::step]` for `step in {1,2,4,8}` — this is stride subsampling, not a resize/pooling downsample), with each scale's x/y finite differences clamped to 100 and mean-reduced over valid pixels, then averaged over the number of valid scales (normally 4):

\[
\mathcal{L}_{grad} = \frac{1}{4}\sum_{s\in\{1,2,4,8\}}\left(\text{mean}|\Delta_x E^{(s)}| + \text{mean}|\Delta_y E^{(s)}|\right).
\]

GeoVR's released `DenseHead.forward` always returns `(depth, None)` — no predicted confidence — so its confidence-weighted term `loss_conf` is always `torch.tensor(0.0)` in the released configuration (dead code path, matches Section 18's decision not to use teacher confidence in this first experiment). The combined loss is therefore exactly:

\[
\boxed{
\mathcal{L}_{depth} = \mathcal{L}_{reg} + \mathcal{L}_{grad}
}
\]

with no additional per-term weighting hyperparameter between `reg` and `grad` in GeoVR's own code (they are added with implicit weight 1 each; `depth_loss_weight`/`lambda_depth` from Section 19/20 is applied once, to the whole `L_depth` sum, matching GeoVR's own `depth_loss = depth_loss_weight * outputs["depth_loss"]` pattern).

Implement this exact formula. Scales:

```yaml
depth_gradient_scales:
  - 1
  - 2
  - 4
  - 8
```

Do not introduce SILog, scale alignment, median normalization, confidence weighting, or extra losses in the first run — the released GeoVR default already omits all of these for the L1 path, so omitting them here is a faithful reproduction, not a simplification.

---

# 16. Positive Depth Output

**Confirmed exactly from the released code — no longer speculative.** `DenseHead.forward` (`models/dense_head.py:345-352`) computes unconstrained depth logits, pixel-shuffles them up to full resolution, and only then applies positivity:

```python
depth_logits = self.proj(fused)
depth_logits = F.pixel_shuffle(depth_logits, self.final_shuffle_factor)
depth_logits = depth_logits.permute(0, 2, 3, 1)
depth = torch.exp(depth_logits)
```

GeoVR's released DenseHead predicts unconstrained depth logits and applies `torch.exp(depth_logits)` **after** the final `proj` + `pixel_shuffle` upsampling step (not before) to obtain strictly positive depth. Port this exact order — logits → pixel_shuffle → permute → `exp` — rather than applying `exp` earlier in the pipeline.

Add tests:

```python
assert torch.isfinite(pred_depth).all()
assert (pred_depth > 0).all()
```

Do the same finite-value checks for teacher pseudo-depth.

---

# 17. Invalid / Extreme Teacher Depth

Inspect the VGGT depth output distribution on the actual JanusVLN R2R + RxR data before full training.

Log:

```text
min
max
mean
median
p01
p05
p50
p95
p99
NaN fraction
Inf fraction
non-positive fraction
```

Do this separately for:

```text
R2R
RxR
```

If the teacher contains invalid values, define a valid-pixel mask.

Do not silently clamp arbitrary ranges without documenting why.

**Correction — these are two distinct mechanisms; do not conflate them:**

1. **This section (17)** is about diagnosing the *raw teacher pseudo-depth distribution itself* (the min/max/percentile/NaN logging above) — a data-quality check on `D_t^{teacher}`, independent of any loss implementation.
2. **GeoVR's actual `0.98` "outlier keep ratio"** (`filter_by_quantile`, ported in Section 15) is a **different** mechanism: it filters the top 2% of *per-pixel L1 regression error* (`|pred - gt|`), i.e. a robust-loss trick applied during training, not a preprocessing filter on raw teacher-depth values. Confirmed directly from `_compute_depth_loss` in `models/qwen3vl_geo.py`.

So: port GeoVR's residual-quantile filter faithfully as part of `L_depth` in Section 15 (that is what `depth_outlier_keep_ratio: 0.98` in Section 19 configures). Separately, if this section's own raw-teacher-depth diagnostics reveal genuinely invalid values (NaN/Inf/non-positive), define an independent valid-pixel mask for those — do not assume GeoVR's residual filter also cleans up bad teacher data, since it does not operate on teacher values at all.

---

# 18. VGGT Confidence

VGGT may also produce a depth confidence map.

For the **first controlled experiment**:

```text
teacher confidence used in loss = NO
teacher confidence logged        = YES
```

Reason:

We first want to answer whether explicit depth itself helps.

Adding confidence weighting introduces another variable.

If pseudo-depth noise later appears to be a limiting factor, confidence-weighted depth can become a separate follow-up ablation.

---

# 19. Initial Configuration

**The existing `configs/spatial_forcing_vln_*.yaml` files are not consumed by any code — confirmed by a repo-wide grep for any `yaml.safe_load`/config-loader reference to them.** They use a decorative key vocabulary (`teacher.teacher_layer`, `spatial_forcing.loss_weight`, `student.spatial_forcing_layer`, ...) that does not mechanically map onto what's actually read at runtime. The real config surface is CLI flags parsed into `ModelArguments`/`DataArguments`/`TrainingArguments` dataclasses (`src/qwen_vl/train/argument.py`) via `transformers.HfArgumentParser`, with values set directly in `scripts/train/train_spatial_forcing_vln*.sh` (e.g. `--sf_teacher_layer 23`).

**Therefore, add the new depth-supervision knobs as new dataclass fields in `src/qwen_vl/train/argument.py`, plus new CLI flags in a new `scripts/train/train_spatial_forcing_vln_v4*.sh`** — mirroring exactly the pattern the `v1` and `v2` branches each already used when they were built on top of this same `main` (each added its own dataclass fields + its own `train_spatial_forcing_vln_v{N}*.sh` script). The fields needed, for example:

```yaml
depth_supervision_enabled: true

depth_loss_weight: 0.05
depth_head_lr: 1.0e-5

depth_student_layers:
  - 7
  - 16
  - 24
  - 32

depth_loss_type: geo_depth
depth_gradient_scales:
  - 1
  - 2
  - 4
  - 8

depth_outlier_keep_ratio: 0.98   # keep ratio for the per-pixel depth REGRESSION ERROR (loss residual quantile filter,
                                  # matching GeoVR's filter_by_quantile) — NOT a filter on raw teacher-depth values.
                                  # See the correction in Section 17.
depth_use_teacher_confidence: false
```

(shown here in yaml-like notation purely for readability — each key above must become a real dataclass field + CLI flag, not a yaml entry). A yaml file mirroring these names may still be added under `configs/` for documentation parity with the existing (also-unused) ones, but must never be mistaken for the actual mechanism, and must not be the only place these settings are defined.

Names can be adjusted to fit existing config conventions.

When:

```yaml
depth_supervision_enabled: false
```

the model must behave exactly as original v0:

- do not instantiate the student depth head,
- do not enable the VGGT depth head,
- do not capture extra Qwen hidden states,
- do not compute depth targets,
- do not create extra optimizer groups,
- do not change checkpoint contents,
- do not add measurable forward overhead.

---

# 20. Loss Weight Calibration

Do **not** blindly copy GeoVR's global `depth_loss_weight = 1.0`.

The magnitude of our navigation/SF losses is different from GeoVR's training setup.

Before full training:

1. Run 50–100 real batches.
2. Log raw:
   - `navigation_loss`
   - `sf_loss`
   - `0.3 * sf_loss`
   - `depth_reg_loss`
   - `depth_grad_loss`
   - total raw `depth_loss`
3. Measure the initial contribution of the depth branch.
4. Select \(\lambda_{depth}\) so depth is meaningful but does not dominate policy learning.

Initial pilot:

\[
\boxed{\lambda_{depth}=0.05}
\]

is acceptable before calibration, but the final value should be justified by measured loss scales.

A reasonable early-training target is approximately:

\[
\lambda_{depth}\mathcal{L}_{depth}
\approx
0.05\text{--}0.15
\]

rather than allowing depth to become the dominant term.

Do not run a large lambda sweep initially.

---

# 21. Optimizer Groups

Preserve the existing v0 learning rates.

Recommended initial grouping:

```text
pretrained Qwen / merger      : existing v0 LR (e.g. 1e-6)
existing SF projector         : existing v0 projector LR (e.g. 1e-5)
new student depth DenseHead   : 1e-5
VGGT aggregator               : frozen
VGGT depth head               : frozen
```

Inspect GeoVR's optimizer setup as a reference.

If GeoVR uses a larger LR for newly initialized geometry heads, document it, but do not automatically copy it if it destabilizes the current v0 recipe.

The random depth head must **not** accidentally train at the slow pretrained-Qwen LR.

Add an optimizer test asserting every depth-head parameter appears exactly once in the intended parameter group.

---

# 22. Gradient Routing Requirements

Run explicit backward tests with only depth loss active.

Required:

```text
student depth head gradient            > 0
selected Qwen decoder representation   > 0
trainable Qwen parameters              > 0, as expected
VGGT aggregator gradient               = 0
VGGT depth-head gradient               = 0
frozen Qwen vision tower gradient      = 0
```

Also verify existing SF behavior remains correct.

When running:

```python
loss = depth_loss
loss.backward()
```

there must be a real gradient path from dense depth supervision into the intended Qwen student representations.

---

# 23. Distributed / Zero-Loss Safety

Ensure DDP/DeepSpeed compatibility.

Depth supervision should normally have one current frame for every sample, so there should not be a zero-history issue like heading supervision.

However, handle malformed/invalid teacher depth safely.

If an entire batch has zero valid depth pixels:

- do not produce a detached scalar zero,
- produce a graph-connected zero if needed for distributed training,
- log the condition,
- preferably fail during validation rather than silently training such batches.

---

# 24. Checkpointing

The new student depth head is training-only but must be checkpointed during training.

Verify:

```text
save checkpoint:
Qwen weights                         yes
SF projector                         yes
student depth head                   yes
frozen VGGT teacher                  no, same policy as v0
```

Resume must restore:

- depth-head parameters,
- optimizer state for depth-head parameter group,
- scheduler state,
- training step,
- all v0 components exactly.

Add a save → load → forward numerical consistency test.

---

# 25. Inference Must Remain Stock-Qwen

At inference/evaluation:

```text
instruction + RGB history/current
              |
            Qwen
              |
            action
```

The following must not run:

```text
VGGT                     NO
VGGT depth head          NO
student depth head       NO
depth pseudo-labels      NO
depth loss               NO
```

Evaluation code for:

```text
v0
v1
v2
v0 + depth
```

should remain identical aside from model/checkpoint path.

If training-only state-dict keys are present, loading behavior must be explicitly tested so stock Qwen inference loads all actual Qwen weights correctly while ignoring auxiliary depth components where intended.

---

# 26. Training Metrics

Log at minimum:

```text
navigation_loss

spatial_forcing_loss
sf_weighted_loss
mean_sf_cosine_similarity

depth_loss
depth_reg_loss
depth_grad_loss
depth_weighted_loss

teacher_depth_min
teacher_depth_max
teacher_depth_mean
teacher_depth_median

pred_depth_min
pred_depth_max
pred_depth_mean
pred_depth_median

depth_mae

current_qwen_token_count
depth_target_height
depth_target_width

depth_head_grad_verified
qwen_depth_grad_verified
vggt_grad_verified_zero
vggt_depth_grad_verified_zero
```

Log the exact arithmetic:

\[
L_{total}
=
L_{nav}
+
0.3L_{SF}
+
\lambda_{depth}L_{depth}.
\]

---

# 27. Qualitative Visualization

Add a diagnostic utility that periodically renders:

```text
+----------------+----------------+----------------+----------------+
| Current RGB    | VGGT pseudo    | Student depth  | Absolute error |
|                | depth          |                |                |
+----------------+----------------+----------------+----------------+
```

This is required before trusting loss curves.

Use the visualization to detect:

- horizontal/vertical flips,
- crop mismatch,
- aspect-ratio mismatch,
- scale explosions,
- constant student outputs,
- edge misalignment,
- invalid pseudo-depth.

Do not start full training until several randomly sampled R2R and RxR examples look spatially aligned.

---

# 28. Tests

Add focused tests covering at least the following.

## 28.1 v0 regression test

With:

```yaml
depth_supervision_enabled: false
```

verify:

```text
existing logits unchanged
existing nav loss unchanged
existing SF loss unchanged
existing teacher feature unchanged
no depth head instantiated
no extra VGGT depth head instantiated
```

within numerical tolerance.

## 28.2 Layer extraction

Verify:

```text
layers = [7,16,24,32]
```

are actually the intended Qwen decoder outputs.

## 28.3 Current-token extraction

Each selected layer must select the exact same current-image region/order.

## 28.4 Layer-24 reuse

The layer-24 tensor used by depth should correspond to the same representation used by the existing v0 SF branch.

## 28.5 Single VGGT backbone forward

One teacher call should generate both:

```text
SF feature
depth pseudo-label
```

without a second aggregator execution.

## 28.6 Frozen teacher

No VGGT parameter may have a gradient.

## 28.7 Depth gradient

Depth loss alone must produce nonzero gradient in:

```text
depth head
intended Qwen student path
```

## 28.8 Spatial shape

Student and teacher depth maps must match H×W exactly.

## 28.9 Finite positive depth

Reject NaN / Inf and verify positivity where required.

## 28.10 Checkpoint round trip

Save and resume with depth supervision enabled.

## 28.11 Teacher-free inference

Verify the trained checkpoint can be evaluated with the existing VLN evaluator without executing VGGT or depth code.

---

# 29. Real-Data Validation Before Training

Run a validation script on actual R2R + RxR samples.

Sample at least:

```text
100 R2R examples
100 RxR examples
```

Check:

- current image identification,
- current-image Qwen token count,
- all selected depth layers,
- VGGT depth shape,
- teacher depth statistics,
- Qwen/VGGT spatial alignment,
- no NaNs/Infs,
- single VGGT aggregator execution.

Produce a short report in the repository `reports/` folder.

Suggested file:

```text
reports/v0_depth_pretraining_validation.md
```

Include example visualizations and numerical statistics.

---

# 30. Smoke Training

After all unit and real-data validations pass:

Run approximately:

```text
100 optimizer steps
```

with the real training pipeline.

Compare directly against the corresponding early v0 log.

Success criteria:

```text
depth loss decreases
depth MAE decreases
predicted depth does not collapse
SF loss remains healthy
navigation loss remains close to v0 behavior
no NaN/Inf
student depth gradients verified
Qwen gradients verified
VGGT gradients zero
```

If navigation loss immediately becomes significantly worse than v0, inspect/adjust the **depth-loss weighting** before changing architecture.

Do not immediately redesign the DenseHead.

---

# 31. Short Pilot Before Full Training

If 100-step smoke is healthy, run a short pilot (for example 2k–4k optimizer steps depending on compute).

Monitor:

```text
nav loss
SF loss
SF cosine
depth loss
depth MAE
depth qualitative outputs
gradient norms
throughput
peak VRAM
```

Evaluate an intermediate checkpoint on the same R2R validation protocol used for v0.

The short pilot should answer:

> Is explicit depth supervision at least neutral/positive for navigation before committing to the full combined R2R+RxR run?

---

# 32. Full Controlled Experiment

Once validated, train:

\[
\boxed{\text{v0 + Depth}}
\]

using the exact same:

- R2R + RxR training annotations,
- Qwen backbone,
- training steps,
- effective batch size,
- random seed where possible,
- optimizer schedule,
- v0 SF settings,
- image history,
- prompts,
- evaluator,
- R2R val-unseen episodes

as the existing v0 experiment.

The only scientific difference should be:

```text
GeoVR-style depth auxiliary supervision enabled
```

---

# 33. Required Comparison

Report:

| Method | SF | Depth | Heading | SR | SPL | OS | NE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT |  |  |  |  |  |  |  |
| v0 | ✓ |  |  | 48.50 | 41.86 | 61.83 | 6.33 |
| v0 + Depth | ✓ | ✓ |  | TBD | TBD | TBD | TBD |
| corrected v2 | ✓ |  | ✓ | TBD | TBD | TBD | TBD |

Do not combine depth + heading until these isolated comparisons are complete.

---

# 34. Interpretation

The depth experiment is useful regardless of outcome.

If:

```text
v0            ~48.5 SR
v0 + depth    ~48–49 SR
```

then explicit static depth provides little improvement beyond VGGT feature distillation.

That supports the hypothesis that **additional static geometry alone is not the missing ingredient**.

If:

```text
v0 + depth    >50–52 SR
```

then explicit physical geometry supervision carries useful information that feature-level SF does not force the VLN policy to internalize sufficiently.

That would motivate carefully studying:

```text
feature geometry vs explicit depth vs trajectory geometry
```

Do not claim depth loss itself as a novel contribution because GeoVR already uses explicit depth supervision.

Use it primarily as:

```text
GeoVR-style explicit-geometry control / ablation
```

unless the final method contains a genuinely new VLN-specific formulation.

---

# 35. Things NOT to Add in This Version

Do not add:

```text
camera loss
scale loss
heading loss
progress loss
causal VGGT teacher
residual memory
relation memory
point-map supervision
track supervision
depth confidence weighting
multiple depth loss variants
new inference modules
```

This version answers one question only:

\[
\boxed{
\text{Does explicit VGGT pseudo-depth improve SpatialForcing v0?}
}
\]

---

# 36. Suggested Implementation Sequence

1. Restore VGGT depth-head source files (`dpt_head.py`, `camera_head.py`, `track_head.py`, `head_act.py`, `utils.py`, and their geometry/pose_enc/rotation helpers) from branch `spatialstack-working` into `main`'s `src/qwen_vl/model/vggt/heads/` — these are currently untracked on `main` (only stray `.pyc` files remain), and `enable_depth=True` cannot work without them.
2. Checkout/start from the verified SpatialForcing v0 code.
3. Inspect the provided GeoVR repository deeply.
4. Write the GeoVR depth code-audit note.
5. Verify Qwen hidden-state indexing.
6. Refactor the VGGT teacher (`VGGTEncoder`) to add a single-pass `encode_features_and_depth`-style method: one `aggregator(...)` call producing both the existing SF layer slice and (via a direct call to `self.vggt.depth_head(...)` on the same `aggregated_tokens_list`, bypassing `VGGT.forward()`) the dense depth/confidence output.
7. Fix current-frame Qwen/VGGT preprocessing alignment in `src/qwen_vl/data/utils.py::prepare_image_inputs`'s `qwen3.5` branch — stop reloading the raw image for the geometry/VGGT input and instead reuse the already-resized/cropped Qwen tensor (as the sibling `use_geometry_encoder` branch already does), gated behind `depth_supervision_enabled`. Then verify alignment with the validation utility from Section 12.
8. Port/adapt the GeoVR DenseHead.
9. Capture Qwen layers `[7,16,24,32]` in one forward.
10. Extract current-frame visual tokens at every selected layer.
11. Implement exact GeoVR-style depth loss based on the provided code.
12. Add config and optimizer plumbing (new dataclass fields in `src/qwen_vl/train/argument.py` + new CLI flags in a new `scripts/train/train_spatial_forcing_vln_v4*.sh`, per Section 19 — not yaml-only).
13. Add gradients, checkpointing, and inference safeguards.
14. Add unit tests.
15. Run R2R/RxR real-data validation.
16. Produce qualitative depth visualizations.
17. Run 100-step training smoke test.
18. Calibrate `depth_loss_weight` from measured loss magnitudes if necessary.
19. Run short pilot.
20. Evaluate.
21. If healthy/promising, launch the full matched R2R+RxR training run.
22. Write a detailed implementation/training report in `reports/`.

---

# 37. Final Deliverables

After implementation, provide:

```text
1. List of modified files.
2. Explanation of every important code change.
3. GeoVR files/functions used as references.
4. Exact final student depth architecture.
5. Exact hidden-state indexing semantics.
6. Exact final depth-loss equation/code behavior.
7. Proof that VGGT backbone runs only once.
8. Proof that teacher parameters receive no gradients.
9. Proof that depth loss backpropagates into Qwen.
10. Qwen/VGGT image-alignment validation.
11. Real R2R/RxR teacher-depth statistics.
12. Qualitative RGB / teacher / student / error visualizations.
13. Unit-test results.
14. 100-step smoke-training metrics.
15. Loss-scale analysis and chosen lambda_depth.
16. Resume/checkpoint validation.
17. Teacher-free inference validation.
18. Full report saved under reports/.
```

Do not launch a long full training job until all correctness checks above pass.

The priority is **100% correct implementation and a clean controlled experiment**, not adding more geometry objectives.
