# v4 — Feature Reuse and Batched Qwen Readers Implementation Plan

**Goal:** reduce repeated work while preserving the v3 policy, objective and temporal gradient graph.
**Architecture:** unchanged Memory64 + Recent4 + Current. Only execution and feature storage change.
**Spec:** [master §7](00_MASTER_PLAN.md), [contracts C5, C7, C8](09_SHARED_CONTRACTS.md).
**Prerequisite:** v3 deterministic reference trace. No new loss, image compression, action head or memory mechanism is required.

## Required computation graph

```text
unique RGB frames → frozen pre-merger features
                         ├──→ current writer inputs → sequential small writer → M_t for every state
                         └──→ live trainable merger → native visual embeddings
                                                     ↓
               construct independent causal Qwen states for the segment
                                                     ↓
                     reader microbatches + exact gradient bridge
                                                     ↓
                    producer backward → gradient sync → optimizer
```

Each action state is still supervised. Eight states in a batch are not one state's FLOPs. Duplicate frozen computation and redundant long visual prompts are separate savings; report them separately.

## Task 1 — Establish the frozen-feature boundary

**Create:** `src/qwen_vl/data/feature_store.py` and `src/qwen_vl/tools/precompute_features.py`.
**Modify:** `models/qwen_adapter.py` to consume either raw images or validated pre-merger features.
**Interface:** `FeatureStore.get(key, encoder_fingerprint) -> FrameFeatures`; `put(...)` writes only immutable frozen features/grids/transform metadata.

- [ ] Compare official visual `last_hidden_state` for an image processed alone and as part of an independent-image list.
- [ ] Confirm the visual backbone is frozen and deterministic under the chosen preprocessing. Save model revision, preprocessing hash, feature order/dtype/grid and source image checksum.
- [ ] Store pre-merger rows in official block order. The trainable merger executes during every live training segment.
- [ ] Cache hits must reject mismatched grids, preprocessing, encoder checkpoint or corrupted files. Writes are atomic.
- [ ] Use bounded RAM/GPU caches and sharded storage rather than millions of tiny files. Record storage bytes and feature-generation cost.

For the first extractor, running the full official visual call under no_grad and discarding its merged output is acceptable. It does not require a fork of the frozen encoder. It is only a precomputation path: the trainable merger used for action training remains live.

**Never persist merged embeddings or instruction embeddings across optimizer updates** while their producing parameters train. Fixed-weight inference is different and may cache the lastR merged views.

## Task 2 — Deduplicate producer work within a segment

**Modify:** `train/episode_trainer.py`, `models/navigation_policy.py`.

Build `unique_frames` from current frames plus recent contexts needed by the segment. Produce one live merged tensor per unique FrameKey. Run the writer sequentially over only the newly arriving current observations, never over old recent views again. Produce all M_t and retain the producer graph.

The segment may include a fresh episode after a reset. Do not deduplicate semantically different episodes merely because file names or local step numbers coincide. Image-file caching may share identical frozen tensors, but recurrent state must remain keyed by episode/instruction.

## Task 3 — Implement direct batched readers before the bridge

**Modify:** `models/qwen_adapter.py`; add `ReaderBatch` assembly helpers to `contracts.py` if needed.

Ordinary batch dimension is the first implementation. Right-padding, image-grid concatenation, shadow positions, current spans and memory spans must be per state. Different states' labels cannot affect each other. Keep Qwen's inter-decision cache disabled.

- [ ] Forward outputs match sequential v3 for reader microbatch sizes1,2,4, including uneven final microbatches.
- [ ] Direct-autograd loss and gradients match with identical state weights, independent of padding and target-token counts.
- [ ] Perturb another example's instruction, memory or label and confirm this example's logits do not change.

Use deterministic tiny-model float32 tests, then real Qwen bf16 tolerances. Batch kernels can change floating-point rounding; unexplained structural mismatches are not a tolerance issue.

## Task 4 — Implement the exact leaf-proxy bridge

**Create:** `src/qwen_vl/train/gradient_bridge.py`.
**Interface:** `ProducerBridge.register(key, tensor) -> leaf_proxy`; `backward_producers()` runs once after all reader losses.

Follow C8. Register original memory outputs and every shared trainable merged-image output. Distinct semantic references to the same final tensor use one proxy. Reader gradients update Qwen/read adapters immediately and accumulate on proxies; final producer backward sends those gradients into the writer, input/text paths and merger.

- [ ] Test direct versus bridged gradients with a shared merger output reused by multiple states.
- [ ] Test a writer level and its descendant level both receiving losses. Do not duplicate final-state/H3 aliases.
- [ ] Test unused proxy outputs, multiple auxiliary loss consumers, short chunks and differently sized reader microbatches.
- [ ] Do not step or zero producer gradients between reader microbatches.
- [ ] Confirm writer and merger parameters actually change after the combined step at a nonzero learning rate.

The implementation cannot wrap only the Qwen reader in DDP and assume producer synchronization follows automatically. The default explicitly synchronized engine performs one reduction after both backward phases. An alternative engine must pass the same gradient/update tests before use.

## Task 5 — Add measured memory and optional execution improvements

**Create:** `src/qwen_vl/tools/benchmark.py`; extend environment/performance reports.

Separate persistent weights/optimizer buffers, saved activations, temporary logits, feature-cache allocations and inter-device synchronization. Use GPU synchronization around timed boundaries. Report cold-cache preprocessing separately from warmed training throughput.

The FP32-parameter reference engine can be memory-expensive. Evaluate non-reentrant checkpointing and bounded reader microbatches first. A PyTorch optimizer-state-sharding backend is an optional, separately named execution configuration, with single-update and resume parity tests; it must not add duplicate gradient reduction. Keep the unsharded tiny-model reference as an oracle.

An optional exact optimization is to apply the unchanged LM head only at predecessor positions of supervised tokens. Implement only after the full-logit path works. It changes neither four-action semantics nor label likelihood; test per-state loss and every parameter gradient against the original path. Do not silently switch to a four-class head or a new token vocabulary.

## Task 6 — Fixed-checkpoint equivalence and cost comparison

Compare the same v3 checkpoint under sequential execution and v4 execution, using the same data ordering and preprocessing. Then compare a small matched training continuation with saved RNG/optimizer state. v4 starts from public Qwen in the independent architecture experiment; this continuation is an execution-parity test, not the main model result.

Report:
- actual action states/second and unique frames/second;
- repeated vision calls avoided, Qwen input tokens/state, microbatch/padding cost;
- GPU-hours including precompute, peak VRAM, disk cache bytes;
- same actor SR/SPL/NE and warmed observation-to-action latency.

At inference, reuse frozen/merged recent features under fixed weights and keep a boundedR buffer. Do not carry Qwen states that were computed under a different memory prefix.

## Completion gate

Direct/sequential/bridge outputs and gradients agree within predeclared tolerances; cache invalidation works; writer and merger remain trainable; resume works; at least one real measured efficiency report exists. If batching does not improve speed on the chosen hardware, report it honestly and retain the simpler path. v6/v7 can use v4's correct execution without requiring a dramatic speedup.

**Reference implementation:** official Qwen feature outputs and PyTorch `autograd.backward` [REF-QVISION, REF-AUTOGRAD]. VPWEM's precomputed-embedding pipeline is a dataflow reference only [REF-VPPIPE].
