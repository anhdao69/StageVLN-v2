# Shared Technical Contracts

**Authority:** [master plan](00_MASTER_PLAN.md). These are proposed interfaces to create inside `src/qwen_vl`; they do not already exist in the uploaded snapshot. Version-specific tasks implement them incrementally. A field is introduced only when its owning version needs it.

## C1. Common configuration and command contract

Create `src/qwen_vl/config.py` with a strict `RunSpec` loader for JSON. Unknown keys fail. Relative paths resolve relative to the repository root, except paths explicitly documented relative to a dataset root. Expand user/environment path syntax before validation. Write the fully resolved specification before loading a model.

```json
{
  "version": "v0_uniform8",
  "model": {"name_or_path": "Qwen/Qwen3.5-4B", "revision": "resolve_once", "attn_implementation": "sdpa"},
  "data": {"dataset_config": "configs/datasets/newton_r2r_uniform8.json", "history": "annotation_uniform8", "recent": 0, "max_samples": -1},
  "memory": {"enabled": false, "slots": 64, "width": 512, "layers": 3, "heads": 8, "ffn_width": 2048},
  "training": {"backend": "hf_zero2", "seed": 42, "epochs": 1, "global_action_states": 32, "reader_microbatch": 1, "tbptt_steps": 8, "lr_language": 1e-6, "lr_merger": 1e-5, "lr_memory": 1e-4, "lr_auxiliary": 1e-5, "weight_decay": 0.01, "warmup_steps": 1, "gradient_checkpointing": false},
  "spatial": {"mode": "none"},
  "runtime": {"output_dir": "output/v0_uniform8", "max_steps": -1, "save_final_model": true}
}
```

`resolve_once` is an instruction to preflight, not a valid completed manifest revision. Resolve the model reference to an immutable Hub revision/local snapshot fingerprint, write that identity, and reject later changes during resume. It is not an invented commit hash.

Define `python -m qwen_vl.train.run --config PATH [--resume CHECKPOINT] [--init-checkpoint CHECKPOINT]`. Resume restores full state; initialization loads weights only and records provenance. They are mutually exclusive. `train/run_experiment.sh` invokes torchrun using that interface. Existing `train/v0_uniform8.sh` becomes a thin compatible wrapper, not a second independent configuration implementation.

The loader validates fixed initial defaults: one-action horizon, unchanged four labels, native current/recent tokens, no previous-action writer input, no hidden auxiliary losses. It also validates `slots == 0` iff memory is disabled internally; serialized unused defaults must not allocate modules. Log which configuration fields are inactive rather than silently allowing an unused switch to affect behavior.

World size, target budget and tail counts are explicit. The launcher can accept deployment overrides for dataset/output/cache paths, model source/revision, worker count, max steps/samples and checkpointing. It must not silently override history, memory or loss coefficients. Print all effective overrides.

## C2. Immutable data and batch records

Add these types to `src/qwen_vl/contracts.py` as they become needed. Tensors carry batch axes only where shown.

```python
from dataclasses import dataclass
from typing import Literal
import torch

Action = Literal["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]

@dataclass(frozen=True)
class FrameKey:
    dataset: str
    episode: str            # includes instruction identity when one route has alternatives
    step: int               # canonical chronological observation position, starts at 0

@dataclass(frozen=True)
class FrameRecord:
    key: FrameKey
    image_path: str         # relative to audited dataset root
    action: Action | None   # None = explicit observation without supervision
    source_record_id: str

@dataclass(frozen=True)
class EpisodeRecord:
    episode: str
    instruction: str
    instruction_sha256: str
    frames: tuple[FrameRecord, ...]

@dataclass(frozen=True)
class ImageSpan:
    key: FrameKey
    start: int              # first image-pad position; excludes vision delimiters
    stop: int               # half-open, before vision-end
    grid_thw: tuple[int, int, int]
    is_current: bool

@dataclass
class TokenizedState:
    input_ids: torch.Tensor       # [L], shadow IDs retained even for inputs_embeds calls
    labels: torch.Tensor          # [L], int64, -100 except complete target suffix
    image_spans: tuple[ImageSpan, ...]
    memory_span: tuple[int, int] | None
    action_start: int             # first target position, or L for a generation prefix
    prompt_length: int
    frame_keys: tuple[FrameKey, ...]

@dataclass
class FrameFeatures:
    key: FrameKey
    premerge: torch.Tensor        # [T*H*W, Dv], official merge-block order, detached
    grid_thw: tuple[int, int, int]
    merged: torch.Tensor | None   # [T*H*W/m², Dtext], live only within current parameter version
    original_hw: tuple[int, int]
    encoder_fingerprint: str

@dataclass
class WriterStep:
    state: torch.Tensor          # [B,64,512]; same object/value as levels[-1]
    levels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

@dataclass
class EpisodeCursor:
    episode: str
    next_step: int
    memory: torch.Tensor | None  # detached CPU FP32 only when serialized
    recent_keys: tuple[FrameKey, ...]
    finished: bool
```

Do not feed metadata, hashes, IDs, labels or file paths as neural features. Batch metadata can carry them for indexing/assertions. Dataset transforms never mutate source records in place after the manifest is written.

`TokenizedState` always has exactly one current image. Positions are computed before padding; the collator records each actual length. Right-pad training examples and build attention masks from those lengths, not from accidental equality with a pad token inside a valid span. Generation starts batch-one unless variable-length generation parity is verified.

## C3. Prompt and target protocol

Create `data/prompting.py` as the single authority. `build_state(record, selected_frames, tokenizer, grids, memory_slots=0, include_target=True) -> TokenizedState` is used by datasets and evaluator. Preserve the audited v0 user wording. Render the assistant non-thinking prefix explicitly and identically in both modes. Do not depend on different tokenizer-default thinking switches.

A saved `prompt_protocol.json` contains system text, assistant prefix, close suffix, action labels, whitespace policy, image delimiters, template hash, tokenizer revision and preprocessing hash. The actor reloads this file. When a compatible tokenizer template is saved too, it must produce the same prefix; structured processor messages are not assumed compatible with a custom string-only template.

For training, all prefix labels are -100; the target is the complete action followed by the declared message close. Keep the currently intended `<|im_end|>\n` close unless a separately recorded v0 decision changes it. Check the action span by comparing token IDs to the expected suffix under the same renderer. Every example has at least one action token. Reject overlength examples; never crop the action or visual sequence silently.

The baseline renderer must pass this real-tokenizer golden assertion for every action and image-count case:

```python
assert torch.equal(training.input_ids[:training.action_start], inference.input_ids)
assert (training.labels[:training.action_start] == -100).all()
assert training.labels[training.action_start:].tolist() == training.input_ids[training.action_start:].tolist()
```

The processor expands images according to actual `image_grid_thw.prod()/merge_size**2`. Check per example, not only the batch sum. Memory placeholders are ordinary non-image IDs with masked labels/modality zero. Their exact assigned positions are recorded; never recover them by searching for every occurrence of the placeholder token elsewhere in the text.

## C4. History and complete episodes

`history_indices(t: int, mode: str, recent: int = 4) -> list[int]` returns indices including current, increasing, unique and ≤t.

```python
import numpy as np

def history_indices(t: int, mode: str, recent: int = 4) -> list[int]:
    if t < 0 or recent < 0:
        raise ValueError("negative time or recent length")
    if mode == "uniform8":
        return list(range(t + 1)) if t <= 8 else np.linspace(0, t, 9, dtype=int).tolist()
    if mode == "recent":
        return list(range(max(0, t - recent), t + 1))
    if mode == "current":
        return [t]
    raise ValueError(mode)
```

At t=100, expected Uniform8 is `[0,12,25,37,50,62,75,87,100]`; Recent4 is `[96,97,98,99,100]`. Do not use `history[-0:]` for Recent0. v0 may keep existing annotation images only after its uniform-rule audit; v1 builds from canonical full episodes.

`build_episode_manifest` must establish actual episode/instruction IDs, frame-order convention and instruction extraction. Prefer original instruction/trajectory metadata. A parser for the verified Janus prompt is acceptable only with anchored fields and round-trip equality; guessing that any quoted substring is the instruction is not. Flag ambiguous duplicates, missing interior current frames, conflicting labels and mixed instructions. Sorting a partial set does not establish completeness.

Select development data by stable hashes of complete episode IDs, with coverage across available scenes. Do not use first-N step records as evidence about recurrent learning.

## C5. Visual feature layout and adapter

For each single image, official pre-merger rows use merge-block ordering. The inverse to a raster grid is:

```python
def premerge_to_raster(rows, grid_thw, merge_size):
    t, h, w = grid_thw
    m = merge_size
    if t != 1 or h % m or w % m or rows.shape[0] != t*h*w:
        raise ValueError("unsupported image grid")
    d = rows.shape[-1]
    return (rows.reshape(t, h//m, w//m, m, m, d)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(t, h, w, d))
```

Write the inverse synthetic permutation test before pooling or teacher alignment. A direct `reshape(H,W,D)` is incorrect for block-major rows. Merged tokens occupy the coarse grid in native order; independently verify that using a coordinate-coded fixture.

Writer pool size: `h2=max(1,round(8*h/max(h,w)))`, `w2=max(1,round(8*w/max(h,w)))`; cap each at its source dimension. Project channels then adaptive-average-pool. No resizing of current/recent native reader tokens is introduced.

`QwenVisualAdapter.encode(images) -> list[FrameFeatures]` uses the official visual output. `QwenReaderAdapter.forward(states, features_by_key, memories, capture_layers=())` builds shadow IDs, images and memory spans, computes official multimodal positions, and calls the model with `inputs_embeds`, explicit positions and no pixels. Pass exactly one of IDs/embeddings. No memory caches or rope deltas from a different batch may influence the call.

For batch-one action decoding, first prefill with assembled prefix embeddings and explicit 3D positions, with a fresh official cache. Predict the first token from the final prefix position. Subsequent calls feed only the newly generated token, using its own `input_ids` (not prefix embeddings), the returned cache, and explicit positions. If L is the unpadded prefix length and p is one plus the maximum prefix 3D position, the j-th token fed after prefill has `cache_position=L+j` and all three text position coordinates `p+j`. Test this against native generation with memory disabled; do not reuse stale model `rope_deltas` from another prompt. The writer is not called during this token loop.

No-memory parity is tested for 1, 2, 5 and 9 images, mixed aspect ratios, unequal padded lengths and a save/reload cycle. Native versus adapter logits, selected hidden states and trainable gradients must agree within declared numerical tolerances. First use deterministic float32/tiny configurations, then the actual bf16 kernels. Record the tolerance rather than changing it until a test passes.

## C6. Pure memory functions and state ownership

`MemoryWriter.initial(batch, device) -> Tensor` returns expanded learnable initial slots. `reset(state, first_mask)` uses differentiable `where` with fresh initial slots, not a detached copy of them. The main trainer/session owns all episode state.

`MemoryWriter.forward(previous, visual_tokens, instruction_tokens, visual_mask, instruction_mask) -> WriterStep` must have no internal cache, labels, timestamps from the dataset, teacher features or previous-Qwen-state input. It is called once per real observation. Numerical step counters are only query metadata for auxiliary supervision, not writer input.

The only temporal detach is at a declared BPTT boundary after all losses have been differentiated. A short episode can end before K; a fresh episode in the slot begins with initial memory and an empty recent buffer. Never allow auxiliary queries across an episode reset even if both episodes occupy the same segment.

## C7. Loss normalization and update ordering

Let S be all real labeled action states contributing to one optimizer update, across ranks and accumulated segments. For each state s with Q_s supervised textual tokens, use:

\[
\ell_s=\frac{1}{Q_s}\sum_{j\in\mathcal A_s}\mathrm{CE}(\text{logits}_{s,j-1}, y_{s,j}),
\quad L=\frac{1}{|S|}\sum_{s\in S}\ell_s.
\]

Labels are shifted exactly once. Different action label token lengths must not change sample weighting when reader microbatch size changes. Spatial terms are also per-state means; add them to `ell_s` before the global state average.

For manual synchronization over W ranks, each rank differentiates `W/N_global * sum(local_state_losses)`. Sum gradients across ranks and divide by W once. The result equals the global action-state mean. Do not additionally divide by K, local microbatch count or accumulation count. All-reduce the real-target count before backward. A rank with no local state still joins the same collectives.

Update order: zero gradients → build complete update schedule/count → for each local segment, rollout, differentiate all of its losses and detach its carried state → synchronize accumulated parameter gradients → global finite check → clip → optimizer step → scheduler step → commit cursors/checkpoint. When several K-step segments contribute to one update, detach between segments after their backward passes even though the optimizer has not stepped. This keeps the gradient horizon K rather than silently extending it to K times the accumulation count. Do not step a shared parameter while its earlier graph is still live.

Default IID accumulation satisfies `W*b*A=32`; default temporal accumulation satisfies `W*slots*K*A_segments=32`. For uneven tails, use the actual count and record it. If a requested hardware arrangement cannot match 32 exactly, fail configuration or explicitly declare a changed target budget; do not silently round.

### Precision and distributed reference engine

For the **correctness reference** temporal engine, store trainable parameters in FP32 and run matrix operations under bf16 autocast. Frozen vision parameters can remain bf16. This makes the small learning-rate update well-defined without quietly relying on bf16-only Adam moments. v0's DeepSpeed mixed-precision path is a different backend, so `_ep` controls must use this same temporal precision/optimizer path.

A full 4B model with replicated FP32 Adam state can have a substantial memory footprint. Profile actual parameter, gradient, optimizer, activation and temporary-logit allocations before a full run. Start tiny-config/one-GPU tests first. Use non-reentrant activation checkpointing and reader microbatching. A separately tested PyTorch optimizer-state sharding backend may reduce persistent optimizer state, but cannot change the math or introduce a second gradient reducer. Do not report this reference engine as the fastest implementation or assert a particular 80GB fit without measurement.

`distributed_grad.py` owns the one synchronization pass. Reduce an active-gradient mask first; for globally active parameters, ranks without a local gradient supply zeros. Leave globally inactive parameter gradients as None. Use deterministic bucket/slice ordering and split very large parameters into bounded-size slices. Reduce FP32 gradients, divide once, then clip. Never enable DDP/DeepSpeed gradient hooks concurrently.

## C8. Exact reader microbatch gradient bridge

The producer graph includes writer outputs, writer instruction features indirectly, and any shared trainable merger outputs. The reader consumes leaf proxies:

```python
# This is the mathematical bridge; production code also handles unused outputs,
# aliasing, numerical checks and the distributed normalization in C7.
originals = tuple(unique_producer_outputs)
proxies = tuple(x.detach().requires_grad_(True) for x in originals)
for reader_batch in reader_batches:
    loss = build_already_globally_scaled_loss(reader_batch, proxies)
    loss.backward()                 # trains reader/aux heads and accumulates proxy grads
used = [(x, p.grad) for x, p in zip(originals, proxies) if p.grad is not None]
if used:
    torch.autograd.backward(tuple(x for x, _ in used), tuple(g for _, g in used))
# Only now: synchronize, clip, optimizer.step(), scheduler.step().
```

Register each original output once by semantic key/tensor identity. Final writer level and `state` are aliases: do not bridge them twice. If a producer output is consumed by several losses, its single proxy accumulates all contributions. An ancestor level H1 and descendant H3 may both be distinct bridge outputs; autograd adds their correct chain-rule contributions in one call.

Do not zero proxy gradients between reader microbatches. Do not recompute dropout/random preprocessing differently between a reference and bridge test. Do not retain reader loss tensors after backward. Step only after the final producer backward. Differentiable producer outputs that are intentionally unused receive no auxiliary gradient; report expected unused branches.

Direct-autograd and bridge gradients must match for the writer, text/input/read adapters, visual merger, Qwen reader and auxiliary heads on a deterministic toy. Repeat on an actual short Qwen segment before long training.

## C9. Geometry coordinate and target contract

Teacher input is deterministic RGB `[0,1]`, letterboxed to square S=518. Let original dimensions be H0,W0 and resized dimensions Hr,Wr after the same recorded rounding. Store exact integer left/top padding. For an original normalized coordinate `(u,v)` measured from image edges:

\[
u_T=(p_x+uW_r)/S,\qquad v_T=(p_y+vH_r)/S.
\]

Query a teacher patch-feature grid with `grid_sample(..., grid=(2*u_T-1,2*v_T-1), align_corners=False)`. Grid cell centers use `(col+0.5)/width,(row+0.5)/height`. Do not use `align_corners=True` with this formula. Exclude positions outside the original resized-image footprint; log edge behavior and validate synthetic coordinate grids.

One image occupies one teacher sequence: `[B,1,3,518,518]`. Derive patch-grid dimensions and `patch_start_idx` from actual teacher outputs. Selected aggregator indices are `(11,17,23)`; verify non-None, feature width and row counts. Targets are raw patch features, detached, with FP32 cosine normalization and epsilon `1e-6`.

Spatial feature targets are privileged training targets but are generated from the same observed RGB images. No metric pose/depth assumption is added. Do not claim they are a physical map. A target fingerprint includes teacher checkpoint, code revision, layer indices, raw-feature convention, preprocessing transform, grid geometry and numeric storage dtype.

## C10. Auxiliary query, averaging and leakage

A memory query contains `(source_step, lag, normalized_uv, level, valid)` in metadata; only `lag` and normalized coordinates form its embedding. Define `enc(s,d) = concat(sin(s*w), cos(s*w))`, with `w_i = 10000**(-i/(d/2))` for i=0,…,d/2−1. The coordinate encoding is `concat(enc(2*pi*u,256), enc(2*pi*v,256))`; the lag encoding is `enc(log1p(lag),512)`. Project each 512-vector to512 and sum. These fixed formulas are a declared starting representation, not a learned timestep/episode lookup. Each level has one decoder block: query self-free cross-attention to the corresponding 64 memory slots, FFN2048 and Linear512→teacher_width. No image/source feature serves as a key.

Current queries have lag0. Retention queries require same episode and `source_step <= current_step - R - 1`, and initially a source inside the live differentiable segment. Sample at most two historical source frames without replacement, then eight valid anchors each; use all eligible sources if fewer than two. Anchors come from the normalized original image area. Sampling is seeded by the run/query RNG, not a neural input ID.

For each state, average query errors separately for write/retain and over levels. Missing retention queries contribute zero, not a divide-by-zero. The overall denominator remains all real supervised action states. This avoids a different effective loss merely because eligible examples were batched together. Report eligible-state fraction and actual delay histogram.

## C11. State, export and evidence files

Checkpoint only at a completed optimizer boundary. Save model/optimizer/scheduler, query/sampler/Python/NumPy/Torch/CUDA RNG, update and supervised-target counts, resolved config, all stream cursors, detached FP32 memory, recent frame keys, episode permutation and rank assignment. No open image handles or live autograd graphs are serialized. Resume checks the input manifest/data fingerprint/world size; changed world size needs an explicit documented replay/repartition path, not silent continuation.

Actor export excludes the geometry teacher and all auxiliary decoders. It contains backbone, memory/read/text modules if enabled, processor/tokenizer, prompt protocol, four-action decoder protocol and model configuration. Validate load/generation in an environment where VGGT is absent.

Every version produces `acceptance.md`, a machine-readable test-results file, `resolved_config.json`, `run_manifest.json`, and relevant data/latency reports. A smoke run passing is not a claimed navigation improvement. Actual unrun GPU or simulator tests remain marked unrun.
