# Review of plans v0–v3 and live main

Reviewed 2026-09-18 against main `d61a5e4`, the installed Qwen3.5 implementation, local R2R metadata/images, and `external/muVLA` / `external/code_vpwem`. An independent reviewer checked architecture and the baseline changes. No training Slurm jobs have been submitted; the user withdrew submission authorization.

## Verdict

| Version | Assessment | Remaining gates |
|---|---|---|
| v0 Uniform8 | The proposed baseline is sound. Live main required targeted fixes before using the plan's objective and data guarantees. Those fixes are now implemented. | Full epoch, checkpoint resume, automatic upload, and simulator evaluation remain unrun. |
| v1 Recent4 | Correct controlled comparison: four preceding observations **plus current**. Implemented by preprocessing the complete verified episodes with the same trainer/labels. | Same full-run/evaluation limitations. Current-only and temporal `_ep` controls remain necessary before attributing v2 improvements to memory. |
| v2 Memory64/current | Coherent architecture, appropriate diagnostic. Ready for staged implementation using the detailed plan, not ready for a full training launch. | Adapter parity, first-update/temporal gradients, distributed tails, deterministic resume, actor parity, and measured K8 full-model memory fit. |
| v3 Memory64/Recent4 | Correct extension of v2: add the recent reader window, keep one writer update per observation. | Requires v2 gates and no-memory R4 parity; recent frame identities survive TBPTT boundaries and reset at episode boundaries. |

## What changed since the uploaded snapshot

The plans describe SDPA/ZeRO-2, batch32, and an older source snapshot. Live main already had FlashAttention2, ZeRO-1, sparse suffix logits and batch2 per GPU. These are execution/recipe differences, not memory architecture differences. This task uses requested global batch64 and keeps the newer native attention/optimizer path after smoke validation. No second experiment-configuration framework was added just to satisfy the original proposed file layout.

The previous README's 631,264 count was inaccurate for this data. Original R2R metadata, all frame directories, and the existing mixed-data annotation agree on **631,244 unique labeled states in 10,819 episodes**. There are no duplicate R2R records. Labels: 404,912 forward, 111,177 left, 104,336 right, 10,819 stop. Maximum episode length is 183 observations.

## v0 fixes and evidence

- Rebuilt R2R-only annotations with relative paths and audited every chronological filename/action against original `train_gt.json.gz`. All regenerated Uniform8 prompts, targets and selected indices match the existing source R2R records. No RxR record enters either configuration.
- Reject overlength examples before any truncation, preventing partial/empty action targets. Validate image-token counts per example and strict two-message roles. Attention masks use actual lengths.
- The loss is a token mean **within each state**, then a mean over actual state exposures in the global accumulation window. This avoids longer action strings getting higher sample weight and handles a short final update. Installed Trainer/Accelerate/DeepSpeed scaling was inspected; there is no extra accumulation or world-size division.
- Exclude normalization vectors and biases from decay; verify unique optimizer coverage. Scope the length sampler to `QwenSFTTrainer`, removing the global Trainer monkeypatch.
- Save the actual non-thinking tokenizer template and `prompt_protocol.json`. All four actions with 1/5/9 images passed exact generation-prefix and tokenizer save/reload checks.
- Prevent implicit resume from arbitrary output checkpoints; require explicit `--resume_from_checkpoint`. Production launchers use a pinned local public-model snapshot and unique output directories.
- Select only actual target-prediction columns for vocabulary logits. The original contiguous suffix could still materialize thousands of unused logits in mixed short/long prompts. A dense-versus-selected loss/gradient test proves the new gathered alignment.
- Standalone launcher defaults to an active environment or local `.venv`. The prepared cluster wrapper explicitly selects the audited shared environment; no environment installation or reference-fork dependency is introduced.

The existing fast path loads bf16 model weights under DeepSpeed with FP32 optimizer master state. The future temporal reference uses FP32 trainable parameters and a different reducer, so matched `_ep` controls are essential.

## v1 implementation and correctness

`prepare_r2r.py` uses original episode instructions plus the complete image sequence, whose count and every action are checked against ground truth. It constructs `max(0,t-4)..t` directly, never the last five Uniform8 samples. At t=100, Uniform8 is `[0,12,25,37,50,62,75,87,100]`, Recent4 is `[96,97,98,99,100]`. Both generated files have identical state IDs/actions/instructions; only image selection and historical image markers change.

The persisted `episodes.jsonl` also supplies v2's chronological source. Each instruction-bearing episode remains distinct. Source duplicate handling is unambiguous here because duplicates are absent; future manifests must reject duplicates or deduplicate consistently across all compared conditions.

## v2 issues resolved in the detailed plan

1. **Full-model memory feasibility:** about 4.206B language parameters plus 27.27M merger parameters make replicated FP32 weights/gradients/Adam state expensive before eight live reader graphs. Reader batch1 alone does not release those graphs. Measure actual allocations after an optimizer update and K8 activation peak. If necessary, validate optimizer-state sharding or explicitly bring forward the C8 gradient bridge; never silently shrink K or freeze the reader.
2. **Instruction encoder ambiguity:** the original plan did not define how eight learned queries compress the instruction. The implementation plan specifies tokenization, length rejection, positional encoding, one pre-LN bidirectional block, then query cross-attention and FFN with masks and exact dimensions.
3. **Unlabeled observations:** K counts observations; unlabeled frames still update state, but not navigation CE. Count real labels for global normalization, and skip optimizer/scheduler updates for a globally label-free schedule. `W*slots*K*A` is a label count only for fully labeled segments.
4. **Live selective-logit alignment:** preserve the explicit gathered prediction/label mapping (or a separately tested equivalent suffix), rather than treating shortened labels as full sequence labels.
5. **Position/cache interfaces:** native prefill/generation and the manual three-axis embedding path must be compared. Use fresh cache per action, explicit positions, `use_cache=False` in training, and no stale model rope deltas.
6. **Parameter ownership:** one backbone owner, existing lexical embeddings detached only on the writer branch, no duplicated registration, preserve the actual checkpoint's tying configuration rather than assuming ties.

The core equations and design are sound: post-observation/pre-action state; native current image; premerge feature detach only on the writer branch; correct merge-block inverse; nonzero read-adapter initialization; differentiable resets; detach only after segment backward; exact C7 reduction; shared-output C8 gradient bridge. They do not prove a navigation benefit.

The external references are appropriately limited. μVLA updates memory through its large backbone; VPWEM mutates/detaches internal query caches. Neither is a drop-in implementation of this external writer. Do not copy those detach behaviors or install their training stacks.

## v3 review

The Recent4 buffer semantics are correct. At t, read previous frame keys plus current, then append current. Keep those keys across K boundaries and clear them with episode reset. If features are cached later, only frozen premerge features may persist unchanged across optimizer updates; trainable merged representations must be recomputed. The writer still consumes only each new observation once.

With K8 and R4, directly trainable off-window delays are 5–7 inside a segment, not arbitrary long-term credit. Forward memory can persist much longer than its gradient horizon. A K16 diagnostic and matched Current-only_ep/Recent4_ep controls are appropriate before strong memory claims. Reset/shuffle probes show dependence, not automatically useful navigation memory.

## Prepared training/storage/upload policy

Selected settings are v0 batch2/GPU × accumulation8 and v1 batch4/GPU × accumulation4, both ZeRO-1 and global64. v0 batch4 without checkpointing OOMed. The longest-instruction/mixed-history GPU checks measured about 65.5GiB (v0) and 71.9GiB (v1) peak PyTorch allocation on rank0. Short-run startup dominates average throughput; these are safe tested candidates, not an exhaustive fastest-backend claim. See `benchmark_results.json`.

Prepared Slurm files request four H10080GB GPUs, **8 CPUs**, one loader worker per rank and one math thread. Weights/checkpoints go to `/groups/yshang/an221229/checkpoints/StageVLN-v2/`. v0/v1 start independently from revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. LR1e-6/1e-5, one epoch, seed42, cosine/one warmup, native image bounds and action objective match.

Accelerate's even-batch padding repeats four states at the end of an epoch: 631,244 unique states, 631,248 exposures, final update16 rather than64. This standard padding is disclosed in each run manifest; loss normalization uses actual exposures. It is not a silent drop or a strict no-repeat episode schedule.

HF upload runs only after successful final saving, requires the completion marker plus safetensors, uploads final model/protocol metadata, and excludes intermediate checkpoints/source snapshots. Private target repositories are `anhdao69/StageVLN-v0-r2r-uniform8` and `anhdao69/StageVLN-v1-r2r-sw4`. Token authentication was checked without exposing the token; it is stored outside the repo with mode0600. No upload has been executed.

## Evidence and limits

Eight unit tests and all nine supplied plan-algebra checks pass. The final four-GPU v0 verification used 76 unique samples (80 padded exposures), exercised the short last accumulation window, exited successfully, and exported a real safetensors model. Reloaded tokenizer/template matched training; merger and language projection weights both changed after the nonzero-LR update. See `verify_v0_tail_export.log` and `export_check.log`.

One earlier exploratory v0 stress shell reported an error after training metrics were written because its launcher was edited during execution. The subsequent clean tail/export run completed with exit0. Prepared jobs now execute from their captured source snapshot, so later working-tree edits cannot alter a running training script. Public-model safetensors were hashed and match their HF cache blob digests (`model_manifest.json`).

See `data_preparation.log`, `source_audit.json`, `unit_tests.log`, `prompt_check.log`, and benchmark/stress logs in this folder. The detailed v2 plan is `v2_memory_only_implementation.md`.

No simulator SR/SPL/NE/OS evaluation, full epoch, end-to-end automatic upload, or v2/v3 implementation was performed. Short finite-loss runs establish training plumbing and resource observations, not 100% correctness or scientific effectiveness. The original plan's full actor/evaluation acceptance gates remain future work; the current task finishes the training code and review without claiming those gates passed.
