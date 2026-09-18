# v5 — Optional Packing Feasibility Implementation Plan

**Goal:** determine whether independent v4 policy states can be packed faster without cross-state leakage or changed gradients.
**Architecture:** unchanged. This version is an optional infrastructure branch, not a research module or prerequisite for spatial supervision.
**Spec:** [master §7.4](00_MASTER_PLAN.md), [contracts C5, C7–C8](09_SHARED_CONTRACTS.md).

## Why this is not ordinary batching

Qwen3.5 has both full-attention layers and recurrent linear-attention/convolution paths [REF-QCFG, REF-QINPUT]. A block-diagonal full-attention mask does not reset the other state. Efficient-VLN's action-isolating trajectory graph also shares a different prefix structure from independent readers that each receive a different memory state [REF-EFF].

Do not copy an Efficient-VLN mask into Qwen3.5 and declare packed training correct. The inspected Efficient-VLN item is a paper reference; no compatible implementation repository is supplied in this package.

## Task 1 — Make a compatibility report before any kernel patch

**Create:** `src/qwen_vl/tools/inspect_packing_support.py`, `tests/integration/test_packing_isolation.py`.
**Input:** installed model class, exact Transformers/PyTorch/kernel versions.
**Output:** `packing_support.json` with supported/unsupported evidence for each layer type.

Inspect whether every attention backend accepts packed sequence boundaries and whether convolution, linear recurrent states, position IDs and loss extraction respect them. Record the actual argument names/signatures found. Absence of a documented/verified boundary path is an unsupported result, not a request to install an unrelated fork.

## Task 2 — Build an independent-batch oracle

Use the v4 ordinary batch path on two variable-length states from different episodes. Preserve separate shadow positions, image grids, memory prefixes and targets. Capture selected logits, hidden states and all trainable gradients.

Required metamorphic tests:

```text
Change state B's pixels/instruction/labels/memory → state A unchanged.
Swap order A,B → outputs follow their own examples.
Add a third padded state → earlier outputs unchanged.
Change A's target suffix → B unchanged.
Reset between flattened segments → equivalent to fresh independent state.
```

Run these for full attention, linear recurrence and convolution separately on a tiny configuration, then end-to-end with real Qwen. Also compare gradient normalization with unequal action label lengths.

## Task 3 — Implement only a verified supported route

If supported, create `src/qwen_vl/models/packed_reader.py` behind `packing.enabled=false` by default. Define explicit sequence-boundary metadata and ensure every layer receives the required reset information. No cross-example generation cache survives.

If support requires custom kernels or a new Transformers fork, stop this version with a written cost/benefit assessment. Do not switch the primary backbone or turn an optional optimization into the critical path. Kernel development is a separate approved engineering project.

## Task 4 — Benchmark only after parity

Compare padding waste, peak memory, targets/second and backward time with the same checkpoint, examples, target count and precision. Include boundary construction and any reorder/unpack overhead. A speedup obtained by changing causal visibility is not an implementation improvement.

## Acceptable completion states

**SUPPORTED:** isolation, loss, gradient, reset and resume tests pass; measured benefit recorded; ordinary batching remains the fallback.

**UNSUPPORTED / NOT WORTHWHILE:** exact missing backend capability or measured overhead is documented. No unsupported path is enabled. Proceed directly to v6 using v4.

This branch is complete when its feasibility question is answered accurately. It does not require a new trained checkpoint or an SR improvement.
