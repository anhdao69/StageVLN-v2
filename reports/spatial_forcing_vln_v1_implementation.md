# v1 causal multi-frame VGGT: implementation and verification report

Date: 2026-08-07<br>Plan implemented: `plans/v1_multiframe_plan.md`<br>Branch: `v1` (from `main`, uncommitted)<br>Status: implemented and verified end-to-end on real R2R data, including a clean 5-step 4-H100 distributed training run

## 1. Result

SpatialStack's Spatial Forcing training path now supports a JanusVLN-style **causal multi-frame VGGT teacher**. v0 supervises Qwen decoder layer 24 with frozen VGGT layer 23 features computed from the current frame alone:

```text
Q_t <-> VGGT(I_t)
```

v1 keeps the QwenVL branch, navigation loss, projector, Spatial Forcing loss, λ, learning rates, batch size, and inference pipeline **completely unchanged**, and replaces only the VGGT teacher's input:

```text
Q_t <-> VGGT_stream(H_1, ..., H_n, I_t)
```

The same 1-9 ordered history+current frames Qwen already sees are streamed through the frozen VGGT-1B aggregator **sequentially, one frame at a time**, carrying its global-attention KV cache across steps (never re-processing earlier frames, never running frames as independent single-frame calls). Only the resulting **current-frame** features are kept and aligned, using the identical VGGT layer, spatial resizing, and token-alignment logic as v0. The cache is local to each training sample and is discarded once `G_current` is extracted, so it never leaks across samples or batches.

VGGT remains strictly training-only. A v1 checkpoint was loaded with the stock `Qwen3_5ForConditionalGeneration` class (same validator used for v0, unmodified) and confirmed to initialize no VGGT, spatial projector, or geometry-fusion module.

A 5-step, 4-H100, DeepSpeed ZeRO-2 smoke run completed cleanly (`train_runtime: 145.8s`, exit code 0). Navigation loss fell from 2.023 to 0.2278 across the 5 steps, and every logged step reported `vggt_multiframe_teacher_enabled=1`, `vggt_has_gradient=0`, `projector_grad_verified=1`, `student_grad_verified=1`, with `teacher_frame_count` varying naturally between 7.25 and 8.5 as real R2R records of different trajectory lengths passed through the batch.

## 2. Design implemented

### Causal KV-cache streaming, ported from JanusVLN into SpatialStack's VGGT

SpatialStack's own `src/qwen_vl/model/vggt/` tree (used for the loss-only teacher) is a from-scratch reimplementation of VGGT that had no KV-cache support and, notably, an added FlashAttention-3 fast path that the vendored `JanusVLN/` reference does not have. Three files were extended, not replaced, so that the new cached code path is strictly additive:

- **`layers/attention.py`**: `Attention.forward` gained `past_key_values`/`use_cache` parameters. When caching, the new key/value is unsqueezed on a frame axis, concatenated with the cached `(k, v)` from every previously streamed frame, and the query's RoPE position grid is tiled across the cached-frame count to match the grown key length (`pos_k.repeat(1, cached_frames, 1)`) before rotary embeddings are applied. The cached branch always uses `F.scaled_dot_product_attention`; it deliberately does not attempt the FlashAttention-3 branch, since that fast path is unvalidated for the asymmetric query/key lengths this cross-frame attention produces, and correctness mattered more here than speed for a loss-only teacher already running under `torch.no_grad()`. The pre-existing FlashAttention-3 branch used by v0's non-cached, single-frame calls is untouched.
- **`layers/block.py`**: `Block.forward` gained the same two parameters and, in the cached branch, calls the attention sub-layer with the cache and adds the residual before running the (stateless) FFN sub-layer normally. The non-cached branch is untouched.
- **`models/aggregator.py`**: `Aggregator.forward` gained `past_key_values`/`use_cache`. Frame-attention blocks are unaffected -- each frame's self-attention over its own patch tokens is independent of every other frame and therefore never needs a cache. Only the global-attention blocks are cache-aware: each call requires exactly one new frame (`S == 1`) for one trajectory (`B == 1`), selects the camera/register "first-frame" or "remaining-frames" token variant based on how many frames have been streamed so far (`past_key_values[0][0].shape[2] + 1` if a cache exists, else `1`), and threads the per-block cache entry through `_process_global_attention`, storing the updated `(k, v)` back into the caller-owned `past_key_values` list.

### Numerical verification of the streaming cache (CPU, no GPU required)

A naive test -- comparing the streaming loop against a **fresh non-cached forward over each causal prefix** -- is not a valid ground truth: non-cached global attention is fully bidirectional within whatever frames are present, so frame 0 in a 2-frame prefix improperly attends to frame 1, contaminating its own hidden state before that state is later used as a cached key/value for frame 1's query. This was discovered empirically (`tests/test_vggt_streaming.py`'s history documents a first attempt against exactly this invalid reference, which mismatched starting at aa-block 1 with growing error through depth, always exactly 0 at aa-block 0 -- the signature of "correct at the point cache is first read, then contaminated by a non-causal reference").

The actual test builds a genuinely causal reference by reusing the real `Block`/`Attention` submodules (same weights as the code under test) with an explicit block-causal mask (`frame_i` may attend only to `frame_j <= frame_i`) applied to every global-attention step. Against that reference, the streaming implementation matches to float32 precision (`max_abs_diff` on the order of `1e-6` to `1e-7`) across 5 frames, 4 aa-block layers, and both the `conv` and `dinov2_vits14_reg` (ViT, matching VGGT-1B's actual patch-embed family) patch-embed variants. `tests/test_vggt_streaming.py` also checks the `use_cache=True, S != 1` and `B != 1` guard, and `VGGTEncoder.encode_layers_streaming`'s shape parity with the plain `encode_layers` call, single-frame degeneracy, cross-call cache isolation, and the `reference_frame != "first"` guard.

### `VGGTEncoder.encode_layers_streaming`

`src/qwen_vl/model/geometry_encoders/vggt_encoder.py` gained a new method that resets `past_key_values = [None] * depth`, loops over the input frames (ordered oldest history frame to current) feeding one `[1, 1, 3, H, W]` frame per aggregator call with `use_cache=True`, and applies the identical layer-selection / spatial-grid reshape / trim / merge / camera-token logic as `encode_layers` to the **final step's** output only. That reshape logic (previously duplicated inline in `encode_layers`) was factored into a shared `_postprocess_patch_tokens` static method so `encode`, `encode_layers`, and `encode_layers_streaming` cannot silently diverge. The method asserts `reference_frame == "first"` (order must stay chronological; the `Qwen3_5ForConditionalGenerationWithSpatialForcing.initialize_spatial_teacher` constructor already hardcodes this) and raises on more than one call producing conflicting frame geometry.

### Data adapter: gathering every history+current frame for the teacher

`src/qwen_vl/data/data_qwen.py`'s `LazySupervisedDataset._get_item` previously prepared a VGGT-ready tensor (`geometry_encoder_inputs`, resized to the Qwen patch grid scaled to VGGT's 14px patch size) only for the current/final frame. With the new `sf_multiframe_teacher` flag, it does so for **every** frame in the record and stacks them in order into `sf_teacher_pixel_values` with shape `[frame_count, 3, H, W]`, raising a `ValueError` if any two frames in the record disagree on that geometry shape (they must, so the aggregator's cache can concatenate matching patch-grid sizes across frames). `frame_count`, `current_image_token_mask`, and `current_image_grid_thw` -- all built from the *final* frame only -- are unchanged. The collator required no changes: `sf_teacher_pixel_values` was already carried as a per-sample list (not a batched tensor), so a `[3, H, W]` v0 entry and a `[frame_count, 3, H, W]` v1 entry are both handled identically.

### `spatial_forcing.py`: selecting the teacher call per sample

`Qwen3_5ForConditionalGenerationWithSpatialForcing._compute_spatial_forcing`'s per-sample loop now branches on `self.sf_multiframe_teacher`: v0 keeps its `teacher_input.shape[0] != 1` check and calls `encode_layers`; v1 asserts `teacher_input.shape[0] == frame_count[batch_index]` (every history+current frame reached the teacher, not a truncated subset) and calls `encode_layers_streaming`. Both branches feed the identical downstream code -- `add_vggt_position_embedding`, `resize_teacher_spatial_grid`, `spatial_forcing_cosine_loss` -- unchanged. Two new scalar metrics, `vggt_multiframe_teacher_enabled` and `teacher_frame_count`, were added alongside the existing `vggt_pos_embed_enabled`-style diagnostics so training logs make the teacher configuration and effective history length auditable per step.

### Configuration plumbing

`sf_multiframe_teacher: bool = False` was added to `ModelArguments` (default preserves v0 behavior byte-for-byte) and threaded through `train_qwen.py` into both the model config and `data_args` (mirroring the existing `spatial_forcing_enabled` pattern, since `data_args`/`model_args` are separate `HfArgumentParser` dataclasses and cannot share a CLI flag name). `Qwen3_5ForConditionalGenerationWithSpatialForcing.__init__` and `LazySupervisedDataset.__init__` both read it via `getattr(..., False)`.

## 3. Files changed

- `src/qwen_vl/model/vggt/layers/attention.py` -- KV-cache streaming attention, FA3 path untouched.
- `src/qwen_vl/model/vggt/layers/block.py` -- cache-aware block wrapper, non-cached path untouched.
- `src/qwen_vl/model/vggt/models/aggregator.py` -- cache-aware aggregator forward and global-attention step, frame-attention untouched.
- `src/qwen_vl/model/geometry_encoders/vggt_encoder.py` -- `encode_layers_streaming`, shared `_postprocess_patch_tokens`/`_autocast_dtype` helpers.
- `src/qwen_vl/model/spatial_forcing.py` -- `sf_multiframe_teacher` config flag, per-sample teacher-call branch, new metrics.
- `src/qwen_vl/data/data_qwen.py` -- multi-frame teacher-input gathering and shape validation.
- `src/qwen_vl/train/argument.py` -- `sf_multiframe_teacher` CLI flag.
- `src/qwen_vl/train/train_qwen.py` -- threads the flag into `config` and `data_args`.
- `README.md` -- v1 section documenting the flag, scripts, and validators.
- `scripts/train/train_spatial_forcing_vln_v1.sh` -- v1 launcher (default recipe identical to v0's, `SF_MULTIFRAME_TEACHER=True`).
- `scripts/train/train_spatial_forcing_vln_v1_full.sh` -- v1 one-epoch R2R+RxR recipe.
- `configs/spatial_forcing_vln_v1_r2r.yaml`, `configs/spatial_forcing_vln_v1_full.yaml` -- human-readable reference configs.
- `scripts/validation/validate_spatial_forcing_multiframe_data.py` -- real-data adapter validator (adds multi-frame-shape assertions to the existing 1/2/4/9-frame check).
- `scripts/validation/validate_spatial_forcing_multiframe_forward.py` -- real Qwen3.5-4B + VGGT-1B forward/backward validator with the multiframe teacher.
- `tests/test_vggt_streaming.py` -- new unit tests (causal-masked-reference equivalence, cache-usage guards, `VGGTEncoder` streaming wrapper behavior).

`scripts/validation/validate_spatial_forcing_data.py`, `validate_spatial_forcing_forward.py`, and `validate_spatial_forcing_inference.py` (the v0 validators) were not modified; the inference validator was reused unmodified in section 6 below to confirm v1 checkpoints are teacher-free at inference, exactly like v0.

## 4. Verification performed

### Unit tests

```text
22 passed (16 pre-existing + 6 new), on both the CPU login node and the 4xH100 compute node
```

The 6 new tests in `tests/test_vggt_streaming.py` cover: streaming-vs-causal-masked-reference equivalence (both `conv` and `dinov2_vits14_reg` patch embeds), the `use_cache` single-frame/single-batch guard, `encode_layers_streaming` matching a plain single-frame `encode_layers` call, its shape parity with a joint multi-frame call, absence of state leakage across repeated calls (proving each call gets a fresh, sample-local cache), and the `reference_frame == "first"` guard.

### Real R2R data-adapter validation

`validate_spatial_forcing_multiframe_data.py` against `/scratch/11528/anhdao69/data/JanusVLN_data/train_r2r.json`, records with 1, 2, 4, and 9 frames:

| Frames | `sf_teacher_pixel_values` shape | Current Qwen grid | Current tokens |
|---:|---|---|---:|
| 1 | `[1, 3, 336, 448]` | `[1, 24, 32]` | 192 |
| 2 | `[2, 3, 336, 448]` | `[1, 24, 32]` | 192 |
| 4 | `[4, 3, 336, 448]` | `[1, 24, 32]` | 192 |
| 9 | `[9, 3, 336, 448]` | `[1, 24, 32]` | 192 |

Every history+current frame reaches the teacher with a matching `336x448` geometry shape, while the Qwen-side current-frame token count is byte-identical to v0's (192) in every case, confirming the QwenVL branch is untouched.

### Real forward/backward pass (single GPU, real Qwen3.5-4B + VGGT-1B)

`validate_spatial_forcing_multiframe_forward.py`, 9-frame record, `sf_use_vggt_pe=True`:

```json
{
  "spatial_forcing_loss": 0.9788050651550293,
  "mean_cosine_similarity": 0.021194910630583763,
  "current_qwen_token_count": 192.0,
  "teacher_token_count": 192.0,
  "teacher_raw_token_count": 768.0,
  "vggt_multiframe_teacher_enabled": 1.0,
  "teacher_frame_count": 9.0,
  "navigation_loss": 1.75627601146698,
  "total_loss": 2.049917459487915,
  "projector_gradient_norm": 0.2408680021762848,
  "projector_grad_verified": true,
  "student_grad_verified": true,
  "vggt_has_gradient": false,
  "max_cuda_memory_gib": 23.24
}
```

`teacher_raw_token_count=768` (the current frame's own `24x32` patch grid) is identical to v0's, confirming only the current frame's features are ever aligned regardless of how many history frames were streamed beforehand.

### v0-equivalence at the degenerate 1-frame case (same process, same weights)

For a record with `frame_count=1` (no history available), `encode_layers_streaming` on that single frame and v0's plain `encode_layers` were both computed within the same loaded model (identical, non-reseeded `spatial_projector` weights) to eliminate cross-run random-init noise as a confound:

```text
v1 (encode_layers_streaming, 1 frame): spatial_forcing_loss=1.0147360563, mean_cosine_similarity=-0.0147359921
v0 (encode_layers, same 1 frame):      spatial_forcing_loss=1.0147360563, mean_cosine_similarity=-0.0147359921
abs loss diff=0.000e+00  abs cosine diff=0.000e+00
```

v1 degenerates to exactly v0 when there is no history to condition on, as required by the plan (a record with no history frames has nothing distinguishing `VGGT_stream(I_t)` from `VGGT(I_t)`).

### History conditioning has a real, measurable effect

For the same 9-frame record, `VGGTEncoder.encode_layers_streaming` was run once over all 9 frames and once over the current frame alone (both through the causal-cache code path, isolating the effect of history from any encode_layers-vs-streaming implementation difference):

```text
full-history vs current-only-streaming: max_abs_diff=61.6250 mean_abs_diff=1.0665 mean_cosine=0.6589
current-only-streaming vs v0 plain (degenerate, should be ~0): max_abs_diff=0.000e+00
```

History frames substantially change `G_current` (mean cosine similarity of only 0.66 between conditioned and unconditioned features -- not noise), confirming the causal KV cache genuinely incorporates trajectory context rather than silently collapsing to the single-frame case for every record.

### Full 4-H100 distributed training smoke test

```text
node:                c561-002 (Slurm job 3381396, partition h100)
launcher:             scripts/train/train_spatial_forcing_vln_v1.sh
world size:           4, per-device batch 1, gradient accumulation 8
dataset:              64-record R2R prefix, MAX_STEPS=5
sf_multiframe_teacher: True, sf_use_vggt_pe: True
train_runtime:        145.8 s (5 optimizer steps; first step included one-time Hopper/Triton kernel compilation, ~112 s, matching v0's documented first-step cost)
```

| Step | navigation_loss | spatial_forcing_loss | total_loss | teacher_frame_count (batch mean) |
|---:|---:|---:|---:|---:|
| 1 | 2.023 | 1.012 | 2.326 | 8.500 |
| 2 | 1.844 | 1.017 | 2.149 | 7.250 |
| 3 | 0.614 | 1.011 | 0.918 | 8.188 |
| 4 | 0.314 | 1.009 | 0.617 | 7.562 |
| 5 | 0.228 | 1.004 | 0.529 | 8.125 |

Every one of the 5 logged steps reported `vggt_multiframe_teacher_enabled=1`, `vggt_pos_embed_enabled=1`, `projector_grad_verified=1`, `student_grad_verified=1`, `vggt_has_gradient=0`, `current_qwen_token_count=192`, `teacher_token_count=192`, `teacher_raw_token_count=768`. `teacher_frame_count` varies step to step because `group_by_modality_length` groups by token length, not history length, so each accumulated batch naturally mixes real R2R records with different trajectory lengths -- exercising the per-sample branch in `_compute_spatial_forcing` under realistic heterogeneous-batch conditions. The run exited with all processes terminating cleanly and the final Hugging Face checkpoint (`model.safetensors`, `config.json`, `trainer_state.json`, etc.) written to `/scratch/11528/anhdao69/spatialstack_runs/sf_v1_smoke_5step_4h100`.

### Teacher-free checkpoint validation (v0's validator, unmodified)

`validate_spatial_forcing_inference.py` against the checkpoint just produced:

```json
{
  "model_class": "Qwen3_5ForConditionalGeneration",
  "forbidden_modules_initialized": [],
  "finite_navigation_loss": 0.1893264353275299,
  "vggt_initialized": false,
  "projector_initialized": false,
  "geometry_fusion_initialized": false
}
```

Stock Transformers reports the six saved `spatial_projector.*` tensors as expected/ignored keys, exactly as in v0; it does not instantiate or execute them. This confirms plan section 6 (v1 inference is identical to v0: no VGGT, no KV cache, no geometry merger) holds for a real trained checkpoint, not just by code inspection.

## 5. Reproduction

From the repository root, on the `v1` branch:

```bash
uv pip install --python .venv/bin/python --no-deps triton==3.7.1   # Hopper only, same as v0

MODEL_PATH=/path/to/Qwen3.5-4B \
TEACHER_MODEL_PATH=/path/to/VGGT-1B \
JANUSVLN_DATA_ROOT=/scratch/11528/anhdao69/data/JanusVLN_data \
CACHE_DIR=/path/to/model-cache \
OUTPUT_DIR=/path/to/output \
NPROC_PER_NODE=4 \
bash scripts/train/train_spatial_forcing_vln_v1.sh
```

For a bounded smoke test, additionally set `MAX_STEPS`, `MAX_SAMPLES`, and `SAVE_STRATEGY=no`, as in this report's section 4 run. For the one-epoch R2R+RxR recipe, use `scripts/train/train_spatial_forcing_vln_v1_full.sh`. "R2R and RxR" here means the combined JanusVLN annotation this repository already ships for v0 (`configs/datasets/janusvln_r2r_rxr.json`); there is no separate RxR-only annotation in this repository, and the combined file's `data_path` (`/mnt/data/vmo-ai-task/anhdh35/JanusVLN`) is not present on this Stampede3 allocation -- point `DATASET_CONFIG` at wherever that annotation actually lives to run it.

To validate the data adapter or run one forward/backward pass without a full training job:

```bash
python scripts/validation/validate_spatial_forcing_multiframe_data.py \
  --model-path /path/to/Qwen3.5-4B --dataset-config configs/datasets/janusvln_r2r.json

python scripts/validation/validate_spatial_forcing_multiframe_forward.py \
  --model-path /path/to/Qwen3.5-4B --teacher-path /path/to/VGGT-1B \
  --data-root /scratch/11528/anhdao69/data/JanusVLN_data --cache-dir /path/to/model-cache
```

## 6. Scope and remaining evaluation

This verifies implementation correctness end to end: data loading, the causal KV-cache mechanics (proved against an independent causal-masked reference, not just "it runs"), forward/backward gradient routing, distributed trainability under realistic heterogeneous per-sample history lengths, and teacher-free inference on a real trained checkpoint. It intentionally used the same VGGT layer, Qwen layer, projector, λ, learning rates, and Qwen history sampler as v0's own smoke configuration, per the plan's "first controlled experiment" instructions -- no hyperparameters were changed alongside the VGGT-input change.

It is not a full R2R navigation-quality experiment: no complete-epoch checkpoint or simulator metrics (SR, SPL, NE) were produced. The next step, per the plan, is a full one-epoch v1 run compared against the existing v0 checkpoint/recipe under the JanusVLN navigation evaluator to answer the plan's primary question -- whether trajectory-conditioned VGGT supervision improves over single-frame VGGT supervision -- before proceeding to memory-length and frame-sampling ablations.
