# v3 — Memory64 + Recent4 + Current Implementation Plan

**Goal:** establish the fixed main architecture before changing execution efficiency or supervision.
**Architecture:** same external writer as v2; the reader additionally receives up to four immediately preceding native image representations.
**Spec:** [master](00_MASTER_PLAN.md), [contracts C2–C8](09_SHARED_CONTRACTS.md).
**Defaults:** M64, R4, K8; same three-block512-wide writer; nav loss only; one-action control.

## What changes from v2

Only the explicit recent window is added. Memory still ingests each observation exactly once and may contain information also visible in the window. Do not reprocess the four old views through the writer at each step. Do not change memory size, gates, input instruction representation, action labels or initial checkpoint.

v3 deliberately keeps native recent-image tokens. Compression of Recent4 would be a new independent design change, not an unreported efficiency tweak.

## Task 1 — Add the recent buffer as episode-owned state

**Modify:** `data/episode_stream.py`, `models/navigation_policy.py`, `eval/session.py`, `train/checkpointing.py`.
**Interface:** `EpisodeCursor.recent_keys` from C2, with at mostR entries before a new step.

- [ ] At t, construct reader frames from previous keys plus current key, in chronological order.
- [ ] Append current only after forming the current state. Drop oldest entries to retainR for the next decision.
- [ ] At a segment boundary, preserve recent frame keys even though memory is detached.
- [ ] At an episode boundary, clear both memory and recent keys. Never reuse features from a same-numbered timestep in a different episode.
- [ ] Training may reload/re-encode old images initially; v4 will optimize duplicate work without changing selection.

```python
def test_recent_buffer_across_segment_boundary(session_fixture):
    s = session_fixture(recent=4, tbptt_steps=8)
    for t in range(9):
        state = s.prepare_dummy_step(t)
        assert state.frame_steps == list(range(max(0, t-4), t+1))
    assert state.frame_steps == [4,5,6,7,8]
```

The fixture implements the proposed preparation/commit lifecycle; include a test that replaying `act` with the same step ID does not update writer state twice.

## Task 2 — Assemble mixed-length visual context correctly

**Modify:** `data/prompting.py`, `models/qwen_adapter.py`.

- [ ] Current-image span remains distinguishable from historical spans for later v6.
- [ ] Every visual span has the correct grid/count and frame identity; reject per-example mismatch even if batch totals agree.
- [ ] Mixed-aspect frames preserve per-image grid geometry; do not assume equalP or one resolution.
- [ ] The memory span remains before the user prompt and has no image modality IDs.
- [ ] No-memory R4 adapter path matches the native v1_sw4 path numerically with identical preprocessing.

## Task 3 — Add the matched Recent4 episode control

**Create:** `configs/experiments/v3_mem64_r4.json`, `v1_sw4_ep.json`.
**Modify:** trainer/state builder to support disabled memory without allocating unused parameters.

Train from the same public checkpoint and the same episode subset/action-state budget. Compare at least Current-only_ep, Recent4_ep, Memory64_R0 and Memory64_R4. This distinguishes benefit from local context, recurrence and chronological training.

## Task 4 — Add memory-use diagnostics before speed work

**Create:** `src/qwen_vl/eval/memory_probes.py`; tests `test_memory_interventions.py`.

Provide controlled switches: reset memory at selected episode steps; freeze updates after a step; replace memory with another episode's state only in a labeled diagnostic; remove the recent window only in a labeled cross-protocol diagnostic. Log intervention time and prior history length.

These interventions test dependence, not automatically correctness: a model can depend on a poor memory. Pair them with closed-loop navigation metrics and length/revisit strata from available metadata. Never inject future observations to manufacture a “memory task.”

## Task 5 — Record the sequential reference trace

Save a small deterministic trace containing selected FrameKeys, image grids, memory norms, gate statistics, prompt/target lengths, losses, selected logits and parameter-gradient checksums. Include a trajectory that crosses a segment boundary and a reset. v4 must reproduce this trace before performance benchmarking.

`acceptance.md` must separate: architecture correctness; tiny overfit; development closed-loop quality; latency/VRAM. Do not treat a smoke loss as evidence that partial observability is solved.

## Completion gate

The R4 selection is exact; state carries and resets correctly; matched controls run; a stable trace exists for v4 parity. Keep M64/R4/K8 fixed. Only one K16 diagnostic is required if memory learning appears weak or before drawing a strong long-memory conclusion. Broader size/window sweeps come after the main pipeline works.

## No new outside code required

Continue the v2 Qwen/μVLA/VPWEM references. Do not add a retrieval bank, memory expert, runtime teacher or backbone-level recurrence. This version is deliberately one coherent state writer plus local context.
