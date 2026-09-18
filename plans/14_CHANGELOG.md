# Revision Changes — Standalone, Code-Grounded Plans

This package replaces the earlier master and v0–v7 plans. It does not modify the supplied training code.

## Scope and independence

- The starting point is the actual13-file plain Qwen SFT snapshot. The namespace remains `qwen_vl`.
- Public pretrained Qwen is the initialization; new navigation/memory modules are trained from scratch. No earlier navigation checkpoint or neighboring environment is an implicit dependency.
- Every version is defined independently of the old paper. Optional StageVLN context is limited to the conceptual geometry-control discussion at v6.
- Version progression is software progression, not an undeclared chain of increasing training exposure.

## v0 corrections promoted into required tasks

- Uniform8 must be audited; the existing loader only checks image count.
- Strict conversation-role and complete-action target validation; reject overlength examples instead of silently truncating.
- One saved train/inference prompt protocol and real-tokenizer round-trip tests.
- Explicit initialization/resume provenance, no automatic checkpoint discovery.
- Declared normalization/bias decay policy, local rather than global Trainer sampler override.
- R2R-only/global32 development recipe is labeled accurately; no claim of joint-data/global64 reproduction.

## Architectural clarification

- μVLA is a reset/TBPTT reference, not a drop-in external updater. Its model and environment are not imported.
- The writer is fully specified:64×512 slots, three gated blocks, instruction-only lightweight conditioning and compact current visual input.
- Qwen's native pre-merger and merged outputs are used before considering a visual-encoder rewrite.
- Feature ordering, multimodal positions, memory placeholders and exactly-one IDs/embeddings behavior are explicit.
- Native Recent4 tokens are retained; no simultaneous recent-image compression or action representation change.

## Training clarification

- Complete episodes and matched no-memory episode controls are required.
- The reference temporal engine explicitly defines precision, per-state loss weighting, once-per-update gradient synchronization and stream checkpoints. Its memory cost is a measurement, not a promise.
- Batched readers preserve producer gradients through an exact bridge covering memory and shared trainable merger outputs.
- Persistent caches stop before the trainable merger; cache identity includes preprocessing and checkpoint.
- Packing is optional and must reset all relevant Qwen states, not only full attention.

## Spatial clarification

- v6 local current-image alignment and v7 memory supervision are separate conditions.
- Teacher targets are explicitly raw single-view VGGT features, not an unspecified inherited positional-augmentation recipe.
- Letterbox/grid coordinate mapping is specified and tested.
- v7 uses memory-only coordinate/lag queries; initial delayed writes stay within the live BPTT segment.
- K8/R4 directly covers off-window lags5–7; long-memory capability is not assumed.
- The endpoint is feature retention, not a calibrated metric map or guaranteed optimal forgetting.

## Agent usability

Added shared contracts, actual codebase map, verified file/symbol reference guide, source collector, explicit evidence gates and copy-ready handoff prompts. Included a hash-tracked unmodified input snapshot so a new agent does not need the previous conversation.
