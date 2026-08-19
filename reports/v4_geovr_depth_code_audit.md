# v4 GeoVR Depth Code Audit

Date: 2026-08-19

This audit was completed before modifying the SpatialStack implementation. It
cross-checks the v4 plan against the supplied GeoVR paper and the local
`GeoVR-MLLM` source tree.

## Sources inspected

- `papers/GeoVR.pdf`, especially Sections 3.2, 3.4, 3.7, 4.1, and Table 6.
  The paper defines a frozen 3D teacher, multi-level MLLM visual-token features,
  a dense decoder, and `L_depth = L1 + gradient`. It reports the released Dense
  Head with L1 supervision as the strongest tested depth-only design.
- `GeoVR-MLLM/models/dense_head.py::DenseHead`. This is the primary reference
  for normalized UV positional encoding, four channel projections
  `[256, 512, 1024, 1024]`, resize factors `[4, 2, 1, 0.5]`, DPT-style feature
  refinement, target-size interpolation, final pixel shuffle, and strictly
  positive `exp(depth_logits)` output.
- `GeoVR-MLLM/models/qwen3vl_geo.py::extract_hidden_states`. GeoVR extracts
  visual-token states from every Hugging Face hidden-state entry before its
  head selects absolute indices.
- `GeoVR-MLLM/models/qwen3vl_geo.py::get_corresponding_qwen_indexes`. The mapping
  is `round(((teacher_index + 1) / teacher_layers) * qwen_layers)`, producing
  `[7, 16, 24, 32]` for teacher indices `[4, 11, 17, 23]` and a 32-block Qwen.
- `GeoVR-MLLM/models/qwen3vl_geo.py::get_geo_features_vggt`. The frozen VGGT
  generates depth/depth confidence and intermediate tokens from its current
  forward result.
- `GeoVR-MLLM/models/qwen3vl_geo.py::filter_by_quantile` and
  `::_compute_depth_loss`. These are the exact references for the robust 98%
  residual filter and four-scale, strided finite-difference loss.
- `GeoVR-MLLM/training/trainer.py::create_optimizer` and
  `GeoVR-MLLM/scripts/train.sh`. GeoVR isolates `dpt_head` in the geometry LR
  group and uses `geo_lr=1e-4`, while its base LR is `2e-5`.
- `GeoVR-MLLM/training/train.py::safe_save_model_for_hf_trainer` and
  `Qwen3VLForConditionalGeneration::load_geometric_weights`. GeoVR checkpoints
  and restores its trainable DPT head while freezing the teacher.

No dedicated automated depth tests were found in the supplied GeoVR repository.
Its useful diagnostics are returned component losses/predictions and the paper's
qualitative depth/point-cloud visualizations. SpatialStack v4 therefore needs
focused tests for the ported head, loss edge cases, gradient routing, the
single-aggregator invariant, checkpoint state, and teacher-free inference.

## Components to port or adapt

1. Port the DenseHead building blocks and math, retaining GeoVR's LayerNorm,
   projection hierarchy, normalized aspect-ratio-aware sine/cosine UV encoding,
   feature fusion, final target-grid interpolation, pixel shuffle, and final
   exponential activation.
2. Adapt `DenseHead.forward` to accept exactly four already-selected tensors in
   `[layer 7, layer 16, layer 24, layer 32]` order. GeoVR indexes a full
   hidden-state tuple by absolute layer number; doing that with SpatialStack's
   four-element list would be incorrect.
3. Port `filter_by_quantile` exactly, including its small-input bypass, detached
   clamped threshold calculation, very-large-input subsampling, `kthvalue`
   indexing, original-tensor mask, and low-survivor fallback.
4. Port the L1 residual and strided gradient loss for steps `[1, 2, 4, 8]`.
   Teacher confidence is diagnostic-only because GeoVR's released DenseHead
   returns no student confidence and the requested controlled ablation excludes
   confidence weighting.
5. Extend the local VGGT wrapper so its aggregator executes once and the same
   `aggregated_tokens_list` feeds both the existing Spatial Forcing layer-23
   slice and the pretrained VGGT depth head.
6. Add a dedicated optimizer group for the student depth head. The v4 plan's
   `1e-5` is intentionally more conservative than GeoVR's `1e-4` because the
   existing VLN recipe trains the pretrained Qwen at `1e-6` and the SF
   projector/merger at `1e-5`.

## Intentional omissions

- Camera tokens/head/loss, metric scale token/head/loss, point maps, tracks,
  teacher confidence weighting, SILog, feature-pyramid distillation beyond the
  existing v0 SF target, streaming/causal VGGT state, and any inference-time
  geometry module.
- GeoVR's full-video treatment. This controlled VLN experiment supervises only
  the final/current observation, matching SpatialForcing v0's teacher scope.
- GeoVR's custom Qwen3-VL model fork. SpatialStack uses the installed
  Qwen3.5 implementation and forward hooks so the stock architecture and
  inference evaluator stay unchanged.

## Qwen3.5/VLN-specific differences

- SpatialStack's visual inputs are separate JanusVLN history/current images, not
  one Qwen video tensor. The existing explicit current-image token mask is the
  source of truth at all four captured layers.
- Hugging Face hidden state `k` denotes the output after decoder block `k` when
  entry 0 is the embedding state. The current hook convention captures decoder
  module `k-1`; v4 must test this equivalence for 7/16/24/32 and keep v0's
  layer-24 hook numerically unchanged.
- The VGGT input must reuse the already resized/cropped current Qwen image when
  depth is enabled. This removes the confirmed field-of-view mismatch while a
  depth-disabled run retains the original v0 preprocessing path.
- The student depth map is resized by the DenseHead to the actual VGGT
  pseudo-depth height/width rather than a fixed GeoVR dataset resolution.
- VGGT remains reproducible and excluded from checkpoints; the student depth
  head is trainable, checkpointed, and absent from stock-Qwen inference.
