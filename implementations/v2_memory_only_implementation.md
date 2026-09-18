# v2 External Memory64 + Current Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement and verify this plan task by task. Do not start a full training run before the hardware and parity gates below pass.

**Goal:** Add an independently updated recurrent visual memory to Qwen3.5-4B, with current-image navigation supervision and a matched current-only episode control.

**Architecture:** A small external writer updates 64 FP32 memory slots once per observation from detached frozen visual features and instruction embeddings. Qwen reads the adapted post-observation memory plus native current-image tokens; its hidden states never update the writer. Chronological training carries state across eight-observation gradient segments.

**Tech stack:** Python 3.12.13, PyTorch 2.10.0+cu129, Transformers 5.3.0, native Qwen3.5 visual/position interfaces. Use plain PyTorch synchronization for the initial temporal correctness backend; do not install the reference repositories' environments.

**Spec:** `plans/00_MASTER_PLAN.md`, `plans/03_v2_memory_only.md`, and contracts C2–C8/C11 in `plans/09_SHARED_CONTRACTS.md`. This document resolves gaps found against live main on 2026-09-18. See `implementations/review_v0_v3.md` for baseline deviations.

## Global constraints and decisions

- R2R only; initialize each primary condition independently from public `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Use the audited complete-episode manifest `/groups/yshang/an221229/data/StageVLN-v2/episodes.jsonl`: 10,819 episodes, 631,244 labeled states. Recheck its SHA256 from `data_audit.json`.
- Requested comparison budget is **64 labeled action states per update**, overriding the original plans' 32. For four ranks and one stream/rank, two fully labeled K8 segments give 64. Count actual labels for tails and unlabeled observations.
- Memory: 64 slots, width 512, 3 blocks, 8 heads, FFN 2048, GELU, dropout 0; recent window R=0; gradient horizon K=8 observations.
- Freeze the visual backbone; train the native merger and language/output parameters. Read Dv/Dtext/merge size from checkpoint configuration (currently 1024/2560/2).
- LR language 1e-6, merger 1e-5, new modules 1e-4; normalization/bias excluded from weight decay 0.01; bf16 autocast with FP32 trainable parameters and Adam state in the temporal reference.
- Native image pixel bounds remain 12544–451584; sequence limit 12800; reject overflow, never crop action targets. Use one textual primitive action and no previous-action features.
- Same seed 42, one epoch, cosine schedule, one warmup update. Define scheduler length from the deterministic update schedule, including actual tails.
- Never carry Qwen generation cache across navigation decisions. No teacher, geometry, action chunking, retrieval, cache optimization, or packed sequences in v2.
- Keep new modules under `src/qwen_vl/`; retain existing v0/v1 SFT entry point. Baseline engine is now audited FA2/ZeRO-1, not the uploaded snapshot's SDPA/ZeRO-2. `_ep` controls must match v2's precision/backend instead.
- Checkpoints and exported weights live under `/groups/yshang/an221229/checkpoints/StageVLN-v2/`. Keep authentication outside configs/checkpoints. Respect the modest CPU budget (initially 8 CPUs on four GPUs).

## Read these concrete interfaces first

1. Installed Transformers `models/qwen3_5/modeling_qwen3_5.py`: `Qwen3_5VisionModel.forward`, `get_image_features`, `get_rope_index`, `Qwen3_5Model.forward`, conditional-generation forward and generation position preparation.
2. Current `data_qwen.py`: custom template, selective logits, image padding and target alignment; `trainer.py`: per-state loss and global target count.
3. `external/muVLA/prismatic/models/memory.py` and `external/muVLA/vla-scripts/finetune.py`: reset/TBPTT examples. Continuing streams must not inherit a default unconditional detach.
4. `external/code_vpwem/cleandiffuser/nn_memory/qformer_memory.py`: attention structure only. Its mutable detached query cache is forbidden in this writer.

Record installed source file hashes and external repository git revisions in `reference_lock.json`. Reference models/action heads are never imported into the policy.

## File and ownership map

| Path under `src/qwen_vl/` | Responsibility |
|---|---|
| `contracts.py` | Frozen frame/episode/span records, feature and writer results |
| `data/history.py`, `data/episode_manifest.py` | Canonical chronology, selectors, source identity |
| `data/prompting.py` | Shared exact text/image/memory span rendering |
| `models/qwen_adapter.py` | One native vision call, feature layout, reader embeddings/positions |
| `models/instruction_encoder.py` | Detached lexical input, trainable instruction compression |
| `models/memory_writer.py`, `models/memory_adapter.py` | Pure recurrent update and read projection |
| `models/navigation_policy.py` | Sole registered backbone owner and new modules |
| `data/episode_stream.py` | Deterministic rank/slot assignment and update schedules |
| `train/losses.py`, `train/optimizer.py` | Shared per-state CE and unique optimizer grouping |
| `train/distributed_grad.py`, `train/episode_trainer.py` | Global normalization and chronological execution |
| `train/checkpointing.py` | Completed-update checkpoints and deterministic resume |
| `eval/session.py`, `eval/action_decoder.py` | Per-episode state, one write per observation, action parsing |
| `tools/profile_temporal.py`, `train/run_episode.py` | Hardware gates and explicit temporal entry point |

## Task 1 — Canonical data, shared prompt, and loss contracts

**Create:** contracts/data/history/data/episode_manifest/data/prompting/train/losses modules above. **Modify:** existing dataset only to call shared rendering without changing serialized v0/v1 examples. **Tests:** `tests/unit/test_episode_manifest.py`, `test_prompting.py`, `test_losses.py`.

**Interfaces:** `load_manifest(path, expected_sha256) -> tuple[EpisodeRecord,...]`; `history_indices(t, mode, recent=4) -> list[int]`; `build_state(record, frames, tokenizer, grids, memory_slots=0, include_target=True) -> TokenizedState`; `navigation_loss_per_state(logits, aligned_labels) -> Tensor[B]`. Use C2 dataclasses verbatim; add explicit `instruction_input_ids` only to a batch-side instruction record, never visual inputs.

- [ ] Write fixtures for t=0/4/8/100, missing step, duplicate step, mismatched instruction, invalid action, and terminal `action=None`. Reject duplicates across all conditions instead of silently changing only one condition's target count.
- [ ] Run `python -m pytest tests/unit/test_episode_manifest.py -q` and observe the missing API failure.
- [ ] Read the audited JSONL, require unique episode IDs, strictly contiguous observation indices, valid relative paths and one stable instruction. Validate action count against image count. Preserve unlabeled views without fabricating STOP.
- [ ] Move the current template into `prompting.py`; serialize its exact hash and action suffix. Implement memory span insertion after system close and before user start, using an ordinary non-image vocabulary ID and recorded half-open positions.
- [ ] Test generation-prefix equality for all four actions and 1/5/9 images; memory disabled removes the entire delimiter/span. No label, path or source ID enters writer text.
- [ ] Implement the CE primitive with an explicit alignment contract: full logits use full labels. The live collator gathers the union of target-prediction positions, prepends an ignored label column and appends an unused logit column; the one-shift CE below remains exact. Preserve that mapping, or use an explicitly aligned contiguous suffix in the adapter. Never shift twice.

```python
# Primitive output is one mean per real action state.
targets = aligned_labels[:, 1:]
valid = targets.ne(-100)
counts = valid.sum(-1)
assert (counts > 0).all()
losses = torch.nn.functional.cross_entropy(
    logits[:, :-1][valid].float(), targets[valid], reduction='none')
rows = valid.nonzero()[:, 0]
per_state = losses.new_zeros(len(counts)).scatter_add(0, rows, losses) / counts
```

- [ ] Verify full/selective loss and gradient equality, padding independence, no-target rejection, and that every image count matches per sample. Run the existing baseline tests as a regression gate. Commit only this tested deliverable.

## Task 2 — No-memory native Qwen adapter parity

**Create:** `models/qwen_adapter.py`, `models/navigation_policy.py`. **Tests:** `tests/unit/test_feature_order.py`, `tests/integration/test_qwen_adapter_parity.py`.

**Interfaces:** `encode(pixel_values, image_grid_thw, keys) -> list[FrameFeatures]`; `read(states, features_by_key, memories=None, use_cache=False) -> ModelOutput`. `NavigationPolicy.backbone` is the only registered owner of Qwen. Adapters are functions or unregistered references; do not register the same backbone again.

- [ ] Test the block-to-raster inverse on a non-square coordinate-coded grid. Each image requires T=1, H/W divisible by merge size and exactly T*H*W premerge rows.

```python
raster = (rows.reshape(t, h//m, w//m, m, m, d)
          .permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, d))
```

- [ ] Call `backbone.model.visual` once for all images. Split `last_hidden_state` by T*H*W and `pooler_output` by T*H*W/m². Detach only the writer's premerge branch; keep merged features differentiable. A whole-vision `no_grad` block would incorrectly freeze merger learning.
- [ ] Get lexical embeddings from the owned backbone, replace only recorded image spans, then compute positions using official `get_rope_index` on shadow IDs, modality IDs, masks and grids. Call the backbone with `inputs_embeds`, explicit `position_ids`, `use_cache=False`, and no input IDs or pixels.
- [ ] Native no-memory and adapter paths must agree in logits, loss, merger gradients and language gradients for 1/2/5/9 images, mixed aspect ratios, unequal padded lengths, and save/reload.
- [ ] Begin with tiny deterministic FP32/SDPA, tolerance `atol=1e-5, rtol=1e-4`; then actual bf16/FA2 on the same inputs with predeclared `atol=2e-2, rtol=2e-2` for selected logits. Report max/relative differences and gradient cosine; investigate structured discrepancies rather than widening tolerance.
- [ ] Assert native embedding/output tying remains exactly as specified by Qwen's actual `tie_word_embeddings` configuration; do not force tying if this revision does not tie them. Verify unique parameter identities in optimizer and state dict. Commit only after parity passes.

## Task 3 — Pure writer, explicit instruction compressor, and read adapter

**Create:** instruction_encoder/memory_writer/memory_adapter. **Tests:** `test_memory_writer.py`, `test_instruction_encoder.py`, `test_memory_gradients.py`.

**Interfaces:** `InstructionEncoder.forward(detached_embeddings, valid_mask) -> Tensor[B,8,512]`; `MemoryWriter.initial(batch,device)`; `MemoryWriter.forward(previous, visual_tokens, instruction_tokens, visual_mask, instruction_mask) -> WriterStep`; `MemoryAdapter.forward(state) -> Tensor[B,64,Dtext]`.

Resolve the previously underspecified instruction encoder as follows:

- Tokenize only actual instruction text with `add_special_tokens=False`, no prompt/action markup; reject empty or >512-token instructions (measure the manifest before fixing this cap for production).
- Detach the existing lexical embeddings only on this branch. Project Dtext→512, add fixed sinusoidal 1D positions, then one pre-LN bidirectional transformer block: eight attention heads, FFN2048/GELU, dropout0, masked padding.
- Eight learned query slots cross-attend to encoded instruction tokens using pre-LN attention, residual and pre-LN FFN2048/GELU residual. Return these eight query states. No second text self-attention stack and no duplicate Qwen embedding table.
- Recompute trainable instruction features once per live segment per distinct instruction, after each parameter update; never persist their old autograd graph across segments. Fixed-weight inference may cache them for the episode.

- [ ] Test masks by changing padded lexical embeddings and asserting unchanged outputs. Test that instruction-projector/query parameters get gradients while the writer branch does not update Qwen's lexical table.
- [ ] For visual writer input, inverse-permute premerge features, project Dv→512, adaptive-average-pool with h2/w2 from C5 (longest side ≤8, never upsample), then add normalized coordinate sine/cosine and a visual type embedding. Native reader tokens remain untouched.
- [ ] Implement three independent blocks exactly as master §6.3. Initial slot parameter is `[64,512]` with `Normal(0,0.02)`. Initialize gate weights 0 and bias -2. Preserve FP32 recurrent state; autocast only matrix operations.

```python
A = H + self_attention(layer_norm(H))
B = A + cross_attention(layer_norm(A), context, context_mask)
proposal = B + ffn(layer_norm(B))
gate = torch.sigmoid(gate_linear(torch.cat([gate_norm(H), gate_norm(proposal)], -1)))
H_next = H.float() + gate.float() * (proposal.float() - H.float())
```

Use separate normalization modules for the attention/FFN/gate sites; never accidentally share them. Context concatenates visual tokens then eight instruction tokens. Preserve three intermediate levels; only H3 recurs.

- [ ] Implement `LayerNorm512→Linear1024→GELU→LinearDtext`, output scale 0.1, normal nonzero final weights. Test initial navigation gradient reaches all writer blocks, input projector and instruction compressor.
- [ ] Run eight toy writes and backpropagate only the final loss; assert nonzero gradient into write0. Repeat with an explicit detach boundary and assert no gradient crosses it. Test future perturbation cannot change earlier returned states, and reset one slot leaves the other untouched.
- [ ] Reject hidden mutable cache attributes. Return a new state rather than editing previous memory in place. Commit after tests pass.

## Task 4 — Chronological scheduler and exact update accounting

**Create:** `data/episode_stream.py`. **Tests:** `tests/unit/test_episode_stream.py`.

**Interface:** `EpisodeScheduler(episodes, seed, rank, world_size, slots=1, K=8, target_budget=64)`; `plan_update() -> UpdateSchedule`; `state_dict()/load_state_dict()`. The global scheduler plan is deterministic on every rank; immutable local image loading can be prefetched separately.

- [ ] Shuffle complete episodes using one recorded permutation, then assign entire streams to ranks. Never use the IID length sampler. Start only at episode boundaries; no empty-state interior chunks.
- [ ] K counts observations, including unlabeled observations. Unlabeled observations still write memory and advance recent state; they produce no action CE. Schedule 64 real labels globally where feasible, with an explicit smaller tail. End a segment at K or an episode boundary; resetting within a segment must still break episode identity.
- [ ] Carry detached FP32 state and frame identities across segments. Reset at episode changes. A short segment does not imply dropping its targets.
- [ ] Test unequal episode lengths, early-exhausted ranks, reset at K boundaries, and unlabeled interior/terminal observations. For globally zero labels, advance observations under no-grad, detach carried state, and skip optimizer/scheduler steps. Do not divide by zero.
- [ ] Save enough cursor/permutation information to reproduce the next schedule exactly. Verify all episode/step pairs are consumed once, with no cross-rank duplication. Commit scheduler independently of GPU code.

## Task 5 — Reference trainer, global gradients, and the matched control

**Create:** train/optimizer/distributed_grad/episode_trainer/run_episode; `configs/experiments/v2_mem64_r0.json`, `v1_current_ep.json`. **Tests:** `tests/distributed/test_temporal_update.py`, `tests/integration/test_temporal_tiny.py`.

**Interfaces:** `build_optimizer(policy,spec)`; `synchronize_gradients(parameters)`; `train_update(schedule, policy, optimizer, scheduler, cursors) -> UpdateMetrics`. Parameters are FP32 except frozen vision, with bf16 autocast for supported compute.

- [ ] Write a two-rank CPU/gloo toy test matching a single-process union of states. Include unequal local targets, one idle rank, unused parameters and two segments per update.
- [ ] Build parameter groups by identity, separating merger/language/new modules and bias/norm decay. Assert exactly one group per trainable parameter.
- [ ] Count global labels **before** backward. For each local segment, writer→reader at every labeled state; sum per-state CE, then backward once through that segment using `world_size/N_global` scaling. Detach carried state immediately after segment backward. Accumulate parameter gradients across segments without stepping or zeroing them.

```python
optimizer.zero_grad(set_to_none=True)
for segment in schedule.local_segments:
    loss_sum, outgoing = rollout_segment(segment)
    (loss_sum * world_size / schedule.global_labels).backward()
    carry = outgoing.detach()
synchronize_gradients(unique_trainable_parameters)  # SUM then /world_size
# All ranks use the same finite decision, clipping, optimizer and LR update.
torch.nn.utils.clip_grad_norm_(unique_trainable_parameters, 1.0)
optimizer.step()
scheduler.step()
```

Handle a segment with no local labels without calling backward on a Python zero. Join identical global collectives even if the rank has exhausted all streams. Reduce an active-gradient mask first; globally inactive parameters stay `grad=None`, locally absent but globally active gradients become zeros. Use deterministic FP32 bounded-size reduction buckets. No DDP/DeepSpeed gradient hooks in this engine.

- [ ] Reduce a global finite flag before stepping, so every rank rejects an invalid update together. Commit cursors only with the completed update/checkpoint boundary.
- [ ] The no-memory `_ep` control allocates no writer and inserts no extra tokens; same scheduler, FP32 optimizer, bf16 compute, native image tokens and 64-label budget. Test its gradient against ordinary per-state training on the identical ordered states.
- [ ] Run a cue-delay toy overfit and a complete real-episode tiny overfit. Report navigation loss and gradient diagnostics; neither is a claim of improved navigation SR/SPL. Commit after mathematical and integration tests pass.

## Task 6 — Mandatory full-model memory and throughput gate

**Create:** `tools/profile_temporal.py`. **Outputs:** `memory_profile.json`, `acceptance.md`, exact command/environment.

- [ ] Measure actual unique trainable parameter count P, frozen bytes, FP32 parameters, gradients and Adam state **after an optimizer update**. A rough replicated lower bound is `16*P` bytes before activations (parameter + gradient + two moments). The actual Qwen language count is about 4.206B, plus merger; do not use “4B” as an exact storage calculation.
- [ ] Run K1, then K8 on the longest audited current-image/instruction state with non-reentrant activation checkpointing. Record allocated/reserved/peak CUDA memory and step time; leave ≥10% GPU headroom for full-data variation.
- [ ] Reader microbatch1 does not release eight retained reader graphs. If replicated FP32 Adam plus K8 does not fit H10080GB, stop full-run scheduling and record the measured allocation.
- [ ] First evaluate a separately tested optimizer-state sharding backend (e.g. `ZeroRedundancyOptimizer` wrapping AdamW), preserving manual gradient synchronization and FP32 moments. Verify checkpoint consolidation/load and one-update parity on the two-rank toy and actual short segment.
- [ ] If activations remain the problem, explicitly move C8's leaf-proxy gradient bridge forward from v4 as a documented execution change. Keep direct K8 as the tiny-model oracle. Do not quietly reduce K, freeze the reader, use bf16-only Adam, or change target normalization.
- [ ] Only after model-gradient parity and actual fit pass, choose runtime and CPU settings from measured throughput and submit a v2 pilot. This plan itself does not authorize implementing v2 during the current v0/v1 task.

## Task 7 — Deterministic resume and export

**Create:** train/checkpointing and tests/integration/test_temporal_resume.py.

**Interface:** `save_update_checkpoint(path, policy, optimizer, scheduler, schedule, cursors, rng, manifest)`; `load_update_checkpoint(path, expected_manifest)`. Use completed optimizer boundaries only.

- [ ] Save all trainable weights, optimizer/scheduler, step and real-label counts, per-rank episode cursors, detached FP32 memory, instruction IDs, Python/NumPy/Torch/CUDA RNG and data permutation. Serialize no live graph or trainable instruction cache.
- [ ] Persist base revision, source manifest SHA256, world size, feature preprocessing, architecture and prompt hashes. Refuse incompatible resume, including changed world size. Fresh initialization and resume are distinct explicit arguments.
- [ ] Compare an uninterrupted two-update toy with save/reload between updates; cursor, loss, parameters and optimizer moments must match. Repeat a short actual-Qwen continuation with documented bf16 tolerances.
- [ ] Export backbone, writer, instruction/read adapters and processor/prompt/action protocols only. No geometry components exist in v2. Validate reload, unique ownership, and forward output before optional HF upload.

## Task 8 — Actor lifecycle and v3 handoff

**Create:** eval/session/action_decoder; tests/test_session_lifecycle.py and integration/test_generation_parity.py.

**Interface:** `PolicySession.reset(episode_id,instruction)`; `observe(step_id,rgb) -> action`. Track observation identity outside neural inputs. Repeated identical step returns the previous result; conflicting RGB/instruction raises. One write per new observation, including turns.

- [ ] Assemble prefix with adapted memory and current native image, then fresh-cache prefill. Generate a single action string, reject invalid strings according to the recorded decoder protocol, and close/discard the Qwen cache after that action.
- [ ] Verify the installed model's manual three-axis position call versus native generation's four-axis preparation (text axis plus three RoPE axes). For continuation use explicit text positions derived from prefix maximum and monotonically increasing cache positions; do not reuse stale rope deltas.
- [ ] Test no-memory manual continuation against native generation, train/inference prefix equality, two interleaved episodes, repeated step, reset, and altered future labels. Labels must not change memory or pre-action states.
- [ ] Record memory reset/freeze diagnostics on available episodes. Closed-loop SR/SPL needs a real simulator adapter and assets; mark it unrun until executed.
- [ ] For v3, add only `recent_keys` with previous four observations. Read before append; retain frame identities/detached frozen features across K boundaries, recompute trainable merger outputs after updates. Compare no-memory R4 against v1_sw4 and create the matched `v1_sw4_ep` control.

## Final acceptance checklist

- [ ] Canonical data identity and exact prompt/target checks pass.
- [ ] Feature ordering, no-memory logits/gradient/generation parity pass.
- [ ] First-update and delayed navigation gradients reach the writer; no cross-boundary or cross-episode leakage.
- [ ] Unequal-tail/idle-rank global loss matches a single-process oracle.
- [ ] Full-model K8 memory fit is measured, including allocated optimizer states.
- [ ] Same-budget current-only episode control exists and trains.
- [ ] Resume and export reload pass; closed-loop metrics are reported only if actually measured.
- [ ] Save machine-readable evidence, resolved configuration and limitations. Do not treat a decreasing smoke loss as proof the memory architecture improves navigation.
