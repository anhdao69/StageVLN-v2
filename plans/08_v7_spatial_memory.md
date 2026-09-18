# v7 — Geometry-Supervised Memory Writing and Retention Implementation Plan

**Goal:** test whether geometry targets improve what the external recurrent state retains after an observation leaves the recent window.
**Architecture:** the same deployed v4 actor. New decoders exist only during training and read writer levels, not Qwen image states.
**Spec:** [master §8.3](00_MASTER_PLAN.md), [contracts C6–C10](09_SHARED_CONTRACTS.md).
**Prerequisite:** v4 gradient bridge and v6 teacher/coordinate adapter. v6's trained weights and current-image loss are not prerequisites.

## The research comparison

v6 asks whether current visible-image representation alignment helps. v7 asks whether the state **after a write and intervening observations** retains geometric feature information. They must use the same teacher/target coordinate protocol and controlled training budgets.

This endpoint predicts historical geometry features. It does not predict metric bearings, depth, poses or a globally calibrated map. Claims about better writing/retention require delayed tests and navigation behavior, not gate visualizations alone. Generic past-feature recall and recurrent memory already have related prior work; the geometry-specific benefit is an empirical hypothesis.

## Task 1 — Expose writer levels without creating extra memory banks

**Modify:** `models/memory_writer.py`, `train/gradient_bridge.py`.

Each write returns H1,H2,H3; only H3 is carried. v7 reads all three for hierarchical supervision. The bridge registers H3 once even when called `state` and `levels[-1]`. H1/H2 may receive independent auxiliary losses and also influence H3; the chain rule must include both paths.

- [ ] With spatial losses disabled, v7 matches v4 output/gradients.
- [ ] A toy where H1 and H3 both receive losses matches direct autograd.
- [ ] No historical teacher target is passed into the writer forward.

## Task 2 — Implement coordinate/lag query decoders

**Create:** `src/qwen_vl/spatial/memory_decoder.py`.
**Interface:** `MemoryGeometryDecoder.forward(memory_level, uv, lag, query_mask) -> predicted_features`.

For each writer level, use a separate decoder with512-wide query embeddings,8-head cross-attention, FFN2048 and an output projection to the corresponding teacher width. Query embeddings use fixed normalized2D coordinates and fixed sinusoidal `log1p(lag)` encoding, each followed by a small learned projection. Queries do not self-attend to teacher features; keys/values come only from the64 memory slots.

No `source_record_id`, episode identifier, image feature, target vector, action text, current Qwen hidden state or true pose enters this decoder. Source-step metadata exists to retrieve a target and compute lag, not as an arbitrary learned episode lookup.

```text
memory H_t^(j) ─────────────→ keys / values
historical coordinate + lag → query
                            ↓
                  predicted teacher feature
                            ↓
          cosine distance to detached target
```

Write a batch-schema test that rejects forbidden neural inputs. A null-memory decoder baseline estimates how much dataset/query regularity can explain.

## Task 3 — Implement causal delayed-query sampling

**Create:** `src/qwen_vl/spatial/query_sampler.py`.
**Inputs:** current frame/episode/step, R, live-segment boundaries, available observed target coordinates, query RNG.
**Outputs:** current writing queries and eligible historical queries with source keys/coordinates/lags.

- [ ] Current writing:8 spatial anchors at tau=t.
- [ ] Retention: at most2 eligible historical observations,8 anchors each, sampled without replacement; use fewer when necessary.
- [ ] A historical source must be in the same episode and satisfy `tau <= t - R - 1`.
- [ ] Initial implementation also requires the source within the current differentiable segment. Never silently treat a detached prior write as receiving long-range gradient.
- [ ] Do not sample unseen future frames or cross episode resets.
- [ ] Query RNG and chosen delays must reproduce on checkpoint resume.

With K8/R4, off-window within-segment delays are5–7. This is an initial controllable training setting, not long-horizon evidence. Log per-step eligibility and actual delays. Run the matched K16 diagnostic before claiming that the same memory learns long-term retention.

Targets generated from full future-inclusive multiview sequences are not allowed in this single-view protocol. The target for tau is computed from the source observation alone.

## Task 4 — Add write/retain losses with stable weighting

For each real supervised state, average each loss over its valid queries and levels. Missing historical queries contribute zero, with all action states retained in the global denominator. This is the explicit C10 convention; do not silently average only eligible states in some microbatches.

\[
L=L_{nav}+\lambda_w L_{write}+\lambda_r L_{retain},
\quad (\lambda_w^{max},\lambda_r^{max})=(0.1,0.2).
\]

Use the same first5%-of-target-budget warmup as v6. Do not add `L_current` by default; a combined objective is a separate named control. Do not add heading/progress or previous-action inputs at this stage.

- [ ] Auxiliary gradients reach the corresponding memory write inside the BPTT segment.
- [ ] Detaching the original write removes that route as expected.
- [ ] Qwen reader microbatch partition does not change write/retain weighting.
- [ ] Duplicate H3 aliases do not double gradients.
- [ ] Removing all teacher/decoder components leaves identical deployed actor outputs at fixed weights.

## Task 5 — Add the indispensable controls

**Create:** configs derived from the same v4 architecture:

| Condition | Auxiliary objective | Interpretation |
|---|---|---|
| `v4_batched` | None | memory infrastructure only |
| `v6_spatial_current` | current Qwen geometry | local supervision control |
| `v7_write_only` | current memory geometry | better writing without delayed retention |
| `v7_retain_only` | delayed memory geometry | delayed objective without write term |
| `v7_spatial_memory` | write + retain | proposed endpoint |
| `v7_visual_recall` | analogous frozen-Qwen-feature recall | generic feature retention control |

For generic recall, pool/resample frozen Qwen features at the same anchors, use the same number/width of decoder blocks and record the changed target dimension/output-layer parameters. An additional fixed projection to a common dimension can be used only when declared and shared across target-family controls. Do not claim exact parameter matching if output dimensions differ.

Also test null/reset memory and a matched sparse-query v6 condition when target density differs. These controls are more important than immediately sweeping64/128 slots.

## Task 6 — Demonstrate useful retention rather than only reconstruction

**Modify:** `eval/memory_probes.py`, metrics reports.

Measure reconstruction/cosine recall against age, especially outside Recent4 and beyond the training truncation length. Report null-memory performance and query eligibility. Use real causal trajectories; avoid unphysical shuffled-frame tests as the only result.

At selected navigation steps, reset/freeze memory and observe decisions/path outcomes, with Recent4 held fixed. Evaluate episode-length and revisit strata when trustworthy metadata is available. A gain should survive comparison against generic recall and current-image alignment. There is no acceptance threshold guaranteeing SR improvement; a negative result is recorded, not repaired with unrelated modules.

## Completion gate

Single-view teacher targets; decoder reads memory only; causal source sampling; gradient bridge and boundary tests; lambda0 equivalence; actor-only export; complete controlled pilot. Only then expand the unchanged experiment protocol to the full audited dataset and targeted memory/window/BPTT ablations.

**End of scope:** this completes the standalone IL-to-spatial-memory project. No later optimization stage is specified or implemented here.
