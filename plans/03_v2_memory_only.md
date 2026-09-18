# v2 — External Memory64 + Current Implementation Plan

**Goal:** prove the recurrent computation and its gradient path without a recent-image bypass.
**Architecture:** `M_t = writer(M_(t-1), frozen_current_features, instruction_features)`; `action = Qwen(instruction, adapted M_t, native current image)`.
**Spec:** [master §5–7](00_MASTER_PLAN.md), [contracts C5–C8](09_SHARED_CONTRACTS.md).
**Defaults:** M64, width512, three blocks, R0, K8, nav loss only. No auxiliary teacher, cache optimization or packed sequences.

## Read-only references before coding

Read the Qwen5.3 visual output and input-embedding interfaces [REF-QVISION, REF-QINPUT]. Read μVLA `MemoryModule`, episodic dataset and TBPTT loop for reset/state scheduling concepts [REF-MUMEM, REF-MUDATA, REF-MUTRAIN]. Read VPWEM's `QFormerLayer` for small self/cross-attention structure [REF-VPQ].

**Do not import their whole models.** μVLA obtains its next memory from backbone processing; our writer must not. VPWEM's detached internal caches are not our gradient policy. Neither repository's environment or action head is a dependency.

## Task 1 — Implement the zero-memory Qwen feature adapter first

**Create:** `src/qwen_vl/models/qwen_adapter.py`, `navigation_policy.py`.
**Interface:** visual `encode(...) -> list[FrameFeatures]`; reader `forward(...)` as C5.

- [ ] One official vision call returns pre-merger rows and merged embeddings. No patched Transformers source and no second visual pass.
- [ ] Extract per-image offsets from actual grids. Invert block order before writer pooling.
- [ ] Assemble text/image embeddings and official multimodal positions. `inputs_embeds` and `input_ids` must not both be passed to the model.
- [ ] With memory disabled, compare native v1 and adapter logits/loss/gradients on deterministic cases before enabling memory.
- [ ] Confirm tied embeddings and unique parameter ownership survive wrapping/saving.

**Tests:** `tests/unit/test_feature_order.py`, `tests/integration/test_qwen_adapter_parity.py`. These are hard gates; poor numerical parity must not be dismissed as memory-training instability.

## Task 2 — Implement the pure writer and instruction pathway

**Create:** `models/instruction_encoder.py`, `memory_writer.py`, `memory_adapter.py`.
**Inputs:** frozen current grid, tokenized instruction, previous state. **Output:** `WriterStep`.

Use the exact master architecture: projected/pool≤64 visual tokens; detached Qwen lexical-embedding branch; instruction block and8 queries; three gated SA/CA/FFN blocks; FP32 carried state. Initialize64 distinct slots and nonzero read adapter. The writer stores no hidden cache and receives no labels or prior Qwen states.

- [ ] Check shapes, masks and absence of NaNs on zero/constant/random visual inputs.
- [ ] Backpropagate a final-step toy loss to the initial slots, early visual projection, instruction encoder and every writer block.
- [ ] Change a future observation and show an earlier returned state is unchanged.
- [ ] Reset one slot without altering another. A reset sends gradient to learned initial slots; a carried detach stops gradient to the previous segment.
- [ ] Confirm the reader can send nonzero navigation gradient into the writer on the first nonzero-LR training update.

```python
def test_late_loss_reaches_early_write(writer_fixture):
    writer, m0, visual, instruction = writer_fixture(length=8)
    early = visual[0].detach().requires_grad_(True)
    state = m0
    for t in range(8):
        out = writer(state, early if t == 0 else visual[t], instruction)
        state = out.state
    state.square().mean().backward()
    assert early.grad is not None and early.grad.abs().sum() > 0
```

The fixture wraps the explicit masks from C6; it does not hide a second updater architecture. Add the complementary boundary-detach test.

## Task 3 — Add memory span assembly and actor lifecycle

**Modify:** `data/prompting.py`, `models/navigation_policy.py`, `eval/session.py`.

- [ ] Insert the masked memory block after the system message, before the original user prompt. Read exact spans, not a global placeholder-ID search.
- [ ] Preserve native current-image tokens. Modality IDs for memory are0; image pads are1.
- [ ] Update memory once per observation, not once per generated action token.
- [ ] Keep per-episode memory independent from Qwen's per-action generation cache.
- [ ] Implement reset and repeated-step idempotence; conflicting image/instruction for an existing step raises.

Verify predicted-prefix equivalence between training and actor. Change the supervised action suffix and show that the writer state and all pre-action reader states are unchanged.

## Task 4 — Add the chronological reference trainer

**Create:** `data/episode_stream.py`, `train/episode_trainer.py`, `train/distributed_grad.py`, `train/checkpointing.py`.
**Modify:** `train/run.py`, optimizer-group helper, configuration.

The scheduler assigns complete episodes to rank-local slots. It yields chronological segments and `first/last/valid` metadata; it does not shuffle steps. I/O workers may prefetch immutable images only.

Within K8, run writer→Qwen sequentially and sum state-normalized losses. Backward through the full live segment before updating. Detached state persists across segments; initial slots replace it only at an episode reset. Do not reuse the IID length sampler.

Implement C7's plain-PyTorch reference backend and explicit one-pass gradient synchronization. Trainable FP32 parameters with bf16 autocast make optimizer behavior explicit. This is a correctness reference with memory costs to measure, not a claim of faster training. Use non-reentrant activation checkpointing if required; do not shrink K silently to fit.

- [ ] Two-rank toy gradient/update matches single-process union-of-states objective, including unequal tails and an idle rank.
- [ ] Globally unused parameters stay grad=None; all rank collectives have identical order.
- [ ] Saved/resumed episode cursors/state/optimizer/RNG reproduce the next toy update.
- [ ] Gradient clipping and scheduler stepping occur once per update, after all local and global backward work.

For a measured full-model OOM, report persistent and activation allocations separately. Do not quietly switch to bf16-only optimizer states or truncate sequences. A validated optimizer-state-sharded execution backend is a separate engineering change; keep reference tests.

## Task 5 — Add matched Current-only episode control

**Create:** `configs/experiments/v2_mem64_r0.json`, `v1_current_ep.json`.

`v1_current_ep` uses the same complete episodes, target count, precision and temporal backend but allocates no writer and inserts no memory span. This is the closest comparator; IID Current-only remains a useful separate training-order control.

Run a tiny synthetic cue-delay task as a plumbing test and a small real episode overfit. Measure memory-reset/shuffle sensitivity as diagnostics, not proof of causal navigation benefit on its own.

## Completion gate

Adapter parity; nonzero early-write gradients; reset/causality tests; two-rank normalization; deterministic resume; actual one-GPU Qwen smoke; same-budget Current-only control. A finite loss alone is insufficient. A recurrent-only SR drop does not automatically block v3, provided the system is correct and the failure is diagnosed.

Do not add a recent window, generic reconstruction loss or geometry to rescue v2 before recording this condition. They belong to explicit subsequent versions.
