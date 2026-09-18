# Experiment and Acceptance Protocol

## 1. Three independent axes

Record separately: **architecture version**, **training condition**, and **execution backend**. v4/v5 primarily change execution. A checkpoint name alone does not identify the supervision, data, precision or initialization.

Example run identity: `v3_mem64_r4__r2r_dev__public_init__nav__ep_fp32_bf16__seed42`. Avoid ambiguous names such as “best memory.”

## 2. Initialization and data budgets

Main conditions initialize from the same pinned public Qwen checkpoint. New modules use a recorded seed. v0→v7 denotes implementation order, not a chain of increasing finetuning exposure. A continuation is allowed only as a labeled debugging/ablation experiment with all prior target exposure counted.

Begin R2R-only because that is the provided dataset config. First define a deterministic development subset from **complete episodes**, using stable ID hashes and scene coverage when reliable metadata exists. Use the same eligible episode set and supervised action states for all relevant conditions. `MAX_SAMPLES=32` is only an entrypoint smoke, not a recurrent learning experiment.

Full-dataset expansion follows the pilot. Adding RxR is a new declared data condition applied consistently, not a hidden improvement to one version. Preserve invalid-record audit counts and never silently drop hard long examples because they exceed memory.

## 3. Minimum comparison ladder

1. Corrected Uniform8 v0; Recent4 v1; Current-only v1.
2. Current-only_ep and Recent4_ep under the temporal training backend.
3. Memory64_R0 and Memory64_R4 under that same backend.
4. Same checkpoint/batches under sequential v3 versus equivalent v4 execution.
5. Geometry-free v4, local-current geometry v6, memory write/retain conditions v7 and generic recall control.

Packing is not a required model comparison. It is an optional equivalent-execution test. Core geometry conclusions do not depend on a packing speedup.

Keep M64/R4/K8 for development. Before broad tuning, test one K16 diagnostic, especially for delayed objectives. Later targeted sweeps may vary slots32/64/128, recent0/2/4 and gradient horizons, but do not run every combination before establishing the main mechanism.

## 4. Stage acceptance evidence

| Gate | Evidence required |
|---|---|
| Data/prompt | certified history rule; complete episode manifest where required; real-tokenizer target/generation prefix equality |
| Backbone integration | native vs assembled-input logits/loss/gradients; image layout and positional-index tests |
| Recurrence | future independence; resets; late-to-early gradient; detach-boundary semantics; actual learned initial-state gradient |
| Optimization | per-state weighting across microbatches/ranks/tails; no duplicate parameters; nonzero-LR weight change |
| v4 execution | shared-feature/memory gradient-bridge equivalence; cache invalidation; checkpoint continuation parity |
| Spatial targets | single-view teacher batching; current/historical coordinate mapping; frozen teacher; no target/label leakage |
| Deployment | actor loads without teacher; correct action parsing; fixed sensor/controller/evaluator contract |

Every report distinguishes passing unit tests, passing real-model integration tests and unrun simulator tests. Skips are not success evidence. No confidence or accuracy number is inferred from a plan or a synthetic test.

## 5. Metrics and accounting

Navigation: SR, SPL, NE and OS, with the exact evaluator protocol and split. Add route-fidelity metrics when available and relevant to the declared dataset. Report per-episode outcomes and multiple seeds for central comparisons when budget allows; paired/bootstrap uncertainty should respect shared scenes/episodes.

Training: real supervised action states/s, unique observations/s, actual Qwen input tokens/state, teacher/inference feature preprocessing cost, total GPU-hours, effective target budget, peak allocated/reserved memory, optimizer-state and feature-cache storage, and data-loader/I/O time.

Inference: batch-one cold and warmed observation-to-action latency p50/p95/p99; vision/writer/prefill/decoding breakdown; policy calls per episode; executed action count; memory footprint versus episode length. Synchronize device timing. Distinguish policy decision from primitive action even though the main protocol executes one action per call.

Caching: disclose cold-cache feature generation, disk space, precision and hit rate. A persistent host archive is still memory consumption. Uniform8 may archive all prior RGB frames to resample; the proposed actor uses bounded recent features plus state. Count host and device storage separately.

Reference temporal backend: explicitly report FP32 trainable parameter storage/bf16 autocast and synchronization/optimizer engine. Do not attribute backend or precision differences to memory architecture. Use `_ep` controls and matched execution comparisons.

## 6. Memory-specific diagnostics

Measure delayed feature recall beyond Recent4 and beyond K, with null-memory/query-only controls. Report lag histograms and eligible-state fraction. The initial K8/R4 loss only directly supervises off-window writes at lags5–7 inside a segment.

Analyze memory reset/freeze effects with local recent observations fixed. Test long episodes, legitimate revisits and instruction transitions when those strata can be derived reliably. Synthetic cue-delay tests establish plumbing; they do not prove real navigation gains. Arbitrary frame scrambling is not a substitute for physically valid partial-observability cases.

## 7. Decision rules

Proceed from v2 to v3 after correctness is established even if recurrent-only performance is weaker. Stop and debug if the reader ignores memory because its gradient path is absent, not merely because an early SR is noisy.

Proceed from v4 to v6/v7 without v5. Spatial supervision is worthwhile only if it improves the relevant representation/behavior relative to generic recall and current-image alignment at declared cost. If it does not, report the result rather than adding unrelated modules until one number improves.

A credible final paper needs a specific retention limitation, a controlled intervention that addresses it, and a navigation quality/cost benefit. These plans do not guarantee a venue outcome.
