# Master Technical Plan — Standalone Qwen Navigation Memory

**Project label:** StageVLN-v2. **Code namespace:** keep `qwen_vl` from the uploaded repository.
**Scope:** expert-demonstration imitation learning, ending at training-only geometric supervision of recurrent memory.
**Status:** proposed architecture and executable implementation roadmap; empirical success is unproven.
**Starting evidence:** [the uploaded code snapshot](inputs/README.md), not an older navigation model.

## 1. Project definition

Build an RGB–instruction navigation policy from a public pretrained Qwen3.5-4B backbone. Progress from an audited explicit-history SFT baseline to a lightweight external recurrent state and a small native-visual recent window. Then test whether geometry targets teach that state to preserve historical visual-spatial information.

“From scratch” concerns the navigation system, new modules, training pipeline, and experiment lineage. The backbone is pretrained. All main comparison conditions start from the same pinned public Qwen checkpoint. A previous StageVLN checkpoint is neither required nor allowed as an undeclared initialization.

**Central hypothesis:** a small memory writer trained independently of per-step Qwen decision states can retain useful trajectory context, and delayed geometric-feature supervision can improve what survives after observations leave the recent window.

The hypothesis is deliberately narrower than metric mapping or optimal forgetting. It does not assert that a feature-reconstruction loss yields accurate 3D localization. This implementation stops at spatial-feature retention; it does not require poses, depth labels, an action-world model, or another training stage.

## 2. What exists versus what must be built

The input repository has a plain conditional-generation SFT path: `R2RSFTDataset`, `DataCollatorForSFT`, `QwenSFTTrainer`, `train_qwen.py`, a length sampler, one R2R dataset configuration, and a ZeRO-2 shell launcher. Its visual backbone is frozen and its merger/language model are trainable.

It does **not** contain a uniform sampler, chronological episode manifest, navigation evaluator, recurrent state, temporal trainer, feature cache, or geometry pipeline. `max_history_frames=8` validates a maximum, not the history distribution. The loader trusts the record's image order.

Retain working code and fix targeted defects before expanding it. Do not rename the package, replace all launchers, or import another repository's training stack without need. The exact path-by-path migration is in [the codebase map](10_CODEBASE_MAP.md).

## 3. Baseline decisions and invariants

### 3.1 Frozen choices

- Backbone initialization: public `Qwen/Qwen3.5-4B`, revision resolved and recorded before training.
- Action vocabulary: `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`.
- Predict one textual action and execute one primitive action per observation.
- Input: RGB observations plus the current episode instruction. No action labels, pose, depth, sample IDs, paths, or teacher features enter the writer.
- Keep native Qwen visual tokens for current and explicit recent frames. Only the writer's own input is pooled.
- Memory development defaults: 64 slots, width 512, three blocks, eight heads, FFN width 2048, dropout 0.
- Recent history default: four **preceding** observations, excluding current.
- Writer BPTT segment: eight observations; forward state persists beyond segments.
- Visual backbone frozen; visual merger and Qwen language/output weights trained. No LoRA switch in the main ladder.
- The geometric teacher and geometric decoders are training-only at v6/v7.
- Persistent Qwen generation/KV state is not carried between navigation decisions. A generation cache can exist within one textual action.

The numeric memory settings and supervision design are project decisions, not claims copied from μVLA. Qwen widths must be read from the actual loaded configuration; the inspected public configuration has visual width 1024 and language width 2560 [REF-QCFG].

### 3.2 Baseline training recipe

Start with the supplied **R2R-only** dataset configuration. Do not silently add RxR or report a joint-data reproduction. Keep the supplied language/merger learning rates (`1e-6` / `1e-5`), cosine schedule, one warmup update, weight decay `0.01`, bf16, TF32, SDPA, one epoch, and the literal current image pixel limits. The default global budget is 32 action states per optimizer update.

For four ranks, the IID baseline uses per-rank batch one and eight accumulation iterations. The recurrent reference uses one stream slot per rank, K=8, and one segment per update. The formulas and short-tail handling are in the contracts. Different hardware changes the accumulation factor, not the stated action-target budget.

Adopt a declared normalization/bias-excluded decay policy at the **new v0** and freeze it thereafter. This is an intentional correction to the supplied custom RMSNorm grouping, not a claim of numerical equivalence with an old run. Archive the raw input snapshot rather than rewriting its history.

Use a per-state token-mean navigation objective, averaged over real action states. This matches the supplied per-device-batch-one objective on valid examples and remains invariant to later reader batching. Verify HF Trainer accumulation behavior explicitly rather than dividing a loss twice.

### 3.3 Standalone environment and ownership

The launcher must use an explicitly activated environment or the repository's own `.venv`; remove the implicit sibling-repository environment dependency. Preserve supplied Python 3.12 / Transformers 5.3.0 / Accelerate 1.13.0 / DeepSpeed 0.16.4 declarations until the real environment is tested. The actual PyTorch/CUDA versions are not established by the attachment: discover, validate, and lock them rather than inventing a compatible version.

A public model reference is resolved once to an immutable revision or an audited local snapshot. Training must fail if an initialization/resume manifest is missing or incompatible. A local path named after Qwen does not prove checkpoint provenance.

## 4. Version ladder

| Version | Policy | Trainer / role |
|---|---|---|
| `v0_uniform8` | Up to eight uniformly selected history frames + current | Corrected standalone SFT and data audit |
| `v1_sw4` | Previous four + current | Isolate explicit-history policy |
| `v1_current` | Current only | Mandatory recurrent-only comparator |
| `v2_mem64_r0` | Memory64 + current | Verify memory and temporal gradient path |
| `v3_mem64_r4` | Memory64 + previous four + current | Fixed main architecture, sequential reader reference |
| `v4_batched` | Exactly v3 | Efficient equivalent execution |
| `v5_pack_optional` | Exactly v3 | Optional kernel/sequence-isolation experiment |
| `v6_spatial_current` | Exactly v3 at deployment | Current-image Qwen-state geometric loss |
| `v7_spatial_memory` | Exactly v3 at deployment | Writer-state current/delayed geometric loss |

`v1_current_ep` and `v1_sw4_ep` are the no-memory configurations evaluated with the same episode scheduler and recurrent training backend. They separate memory from changes in sample order, precision, optimizer implementation, or target reduction.

No version has to improve monotonically. v2 is a diagnostic; v4 must first match v3 numerically; v5 may be rejected without blocking v6/v7.

## 5. Data model and temporal semantics

Use a canonical episode with ordered observations `o_0,…,o_(T-1)` and aligned expert labels `a_0,…,a_(T-1)`. A terminal image without a label is marked unsupervised; it is not assigned a fabricated STOP target. Physical frame order and label alignment must be recovered from metadata or verified filename conventions.

The external writer's state before step t summarizes observations through t−1. At step t:

\[
F_t = E_{\mathrm{frozen}}(o_t),\qquad
V_t = P_{\mathrm{merge}}(F_t),\qquad
C_t = P_W(F_t),
\]
\[
M_t=f_\phi(M_{t-1}, C_t,U_I),
\]
\[
a_t\sim\pi_\theta\big(I,P_M(M_t),V_{\max(0,t-R)},\ldots,V_t\big).
\]

M_t is post-observation and pre-action. The writer runs once per newly received observation, including turn-only transitions. It never receives the expert label a_t. The recent buffer is read before appending current; after the decision it retains at most R observations for the next decision.

This is an **every-observation writer**, not VPWEM's exact out-of-window compressor. Information in memory and the recent window can overlap. Do not label the state an exclusive archive of old views.

### 5.1 Uniform8 and Recent4 are different selection rules

When t≤8, Uniform8 uses all positions 0…t. Otherwise it uses the nine positions given by the validated Janus-style `linspace(0,t,9,dtype=int)` convention, with current at the end [REF-JANUS]. v0 audits the existing annotations instead of assuming this rule was used locally.

Recent4 uses positions `max(0,t-4)…t`. It is **not** the final five images of a Uniform8 record. Build a complete episode manifest from every record's current image or explicit trajectory metadata; verify completeness and ordering before constructing windows. Do not derive episode identity or temporal order from JSON iteration order.

## 6. Architecture in implementation detail

### 6.1 Reuse the official visual outputs; do not rewrite Qwen first

The inspected Transformers 5.3.0 `Qwen3_5VisionModel.forward` returns a `BaseModelOutputWithPooling` with `last_hidden_state` before the trainable merger and `pooler_output` after it [REF-QVISION]. This is sufficient for the first external-memory implementation.

In v2/v3, one native visual call supplies both outputs. Feed detached pre-merger features to the writer and retain the merged-feature graph for action learning. Freezing weights does not justify putting the entire visual call under `no_grad`, because it includes the trainable merger.

Before reshaping pre-merger features into an image grid, reverse Qwen's merge-block ordering. For T=1, hidden rows correspond to `[H/m,W/m,m,m,D]`; permute block/intrablock axes back to `[H,W,D]`. A concrete inverse and tests are in the contracts. The merged output is the native merged-grid sequence. Preserve all grids and per-image offsets.

The supplied path is **multiple independent images**, not a video input. Assert each image grid has temporal count one and verify joint-list versus separate-image parity. A temporal patch size of two does not, by itself, mean two different navigation observations should be combined. Unsupported video input fails instead of being silently reinterpreted.

### 6.2 Writer input and instruction pathway

For each current frozen feature grid, project 1024→512 and pool to a grid whose longest side is at most eight while approximately preserving aspect ratio. This produces at most 64 writer tokens. Add fixed normalized-coordinate 2D sine/cosine encoding and a visual type embedding. Padded cells are masked.

For the instruction, tokenize only the actual instruction text. Obtain lexical embeddings from the existing Qwen embedding table and detach this **writer branch only**. Use a 2560→512 projection, fixed 1D positions, one bidirectional text block, and eight learned query slots to form U_I. Qwen's table remains trainable through its ordinary reader branch. Do not register a duplicate embedding table.

Recompute U_I for each distinct active instruction within every live training segment; never persist a trainable representation across parameter updates. At fixed-weight inference it can be cached for the episode. No per-step Qwen forward is needed to obtain U_I.

### 6.3 Three-block external gated writer

State shape: `[B,64,512]`. The first block receives M_(t−1). For each block, form self-attention and cross-attention residuals, followed by a feed-forward proposal:

\[
A=H+\operatorname{SA}(\operatorname{LN}(H)),
\]
\[
B=A+\operatorname{CA}(\operatorname{LN}(A),[C_t;U_I]),
\]
\[
\widetilde H=B+\operatorname{FFN}(\operatorname{LN}(B)),\quad
g=\sigma(W_g[\operatorname{LN}(H);\operatorname{LN}(\widetilde H)]+b_g),
\]
\[
H'=H+g\odot(\widetilde H-H).
\]

Use eight heads, FFN width 2048, GELU, dropout zero. Initialize distinct memory slots with a small normal distribution, gate weights to zero and gate bias to −2. The gate starts nonzero; it is not supervised by a ground-truth forgetting label. Keep the recurrent residual/state in FP32 and cast attention/MLP computations under bf16 autocast where supported.

Return `WriterStep(state=H3, levels=(H1,H2,H3))`. Only H3 is recurrent. H1/H2 are intermediate states of the same write, not separate memory banks. The function is pure with respect to episode state: no hidden `self.cache` that silently mixes different streams.

μVLA supplies useful reset/BPTT patterns, but its recurrent output comes from the large backbone. That is **not** this update graph. VPWEM's Q-Former supplies a structural attention reference, but its internal detached caches are **not** this BPTT implementation [REF-MUMEM, REF-MUTRAIN, REF-VPQ].

### 6.4 Memory read adapter and Qwen input assembly

Map memory with `LayerNorm(512) → Linear(512,1024) → GELU → Linear(1024,d_text)`, multiplied by a fixed initial scale 0.1. The final layer must not be all zeros, or the initial gradient into the writer can vanish.

At the token level, preserve the v0 user text and image order, but insert a masked memory span after the system message and before the user message:

```text
system message
Memory state:\n [64 non-image placeholder positions] \n
user message with history, current image, and instruction
assistant non-thinking prefix
supervised textual action + message close (training only)
```

The placeholder IDs are existing ordinary vocabulary IDs used only to allocate positions; their embeddings are replaced by memory vectors. No vocabulary resize is needed. They are not `<|image_pad|>` tokens and have modality type zero. The inserted delimiter and memory span have label −100. Memory disabled means the **entire inserted span is absent**, not zero vectors with extra positions.

Assemble token embeddings; replace the recorded image spans with native merged features and the recorded memory span with P_M(M_t). Compute multimodal positions from the unmodified shadow token IDs, modality IDs, masks, and image grids using the official `get_rope_index` helper [REF-QINPUT]. Then call the backbone with `inputs_embeds` and explicit `position_ids`; do not also pass `input_ids`, and do not pass pixels that would cause a second visual pass.

At v2, an empty-memory adapter path must match the v1 native model's logits and gradients before recurrent training begins. This is a hard gate.

### 6.5 Parameter ownership

`NavigationPolicy` owns one `Qwen3_5ForConditionalGeneration` plus the new writer/text/read modules. Helpers reference existing backbone submodules without registering the same trainable module multiple times. Keep Qwen's embedding/output weight tying. Optimizer groups must cover each trainable Parameter object exactly once.

New writer, writer-input projector, instruction encoder and read adapter start at learning rate `1e-4`. Original Qwen language/head and merger keep `1e-6` and `1e-5`. These are starting settings, not tuned claims. Auxiliary spatial heads start at `1e-5`.

## 7. Training architecture and efficiency

### 7.1 Reference temporal trainer

Do not route stateful episodes through the IID length-group sampler or a standard shuffling dataloader. Use a main-process episode-slot scheduler with chronological streams and explicit first/last masks. Image I/O workers may prefetch immutable observations; they never own model state.

Initially use one stream slot per rank and K=8. Run writer → reader at each step, collect correctly normalized losses, backpropagate once per segment, and detach only the carried state after that segment’s backward. If several segments accumulate into one optimizer update, detach between those segments as well, without zeroing accumulated parameter gradients. Do not update optimizer parameters while a live temporal graph still refers to them.

The default reference backend for v2–v7 is **plain PyTorch with explicit once-per-update gradient synchronization**, FP32 trainable parameters and bf16 autocast. Its precision differs from the baseline engine and is matched by the `_ep` controls. This avoids hiding the later two-phase gradient bridge inside unsupported DDP/ZeRO hooks. It is intentionally simple and replicated, not an asserted optimal distributed engine. v0/v1 retain the supplied HF/ZeRO-2 backend; paired `_ep` controls use the new backend. An optimized distributed backend is permitted only as a separately parity-tested execution improvement.

The synchronization contract covers unequal local target counts, a rank that exhausts its episodes first, globally unused parameters, bf16/FP32 buckets, clipping and update ordering. See the contracts and v2/v4 plans. Never combine manual all-reduce with DDP/DeepSpeed reduction.

### 7.2 Carry and truncation

The state remains through an episode; the gradient graph lasts K observations. K=8 is a development setting, not proof that long-term credit assignment is solved. In a segment of eight positions, the maximum within-segment historical lag is seven. With R=4, directly trainable off-window lags are 5–7.

A memory loss beyond a detach boundary can improve later preservation/reading, but cannot train the original write through that boundary. Measure both forward-state horizon and gradient horizon. Run a K=16 diagnostic before treating failure as evidence against memory, and evaluate recall beyond the trained delays.

If a stream starts at an interior observation, replay the actual prefix without gradients to construct an incoming state. Empty-state random interior chunks are forbidden in the default pipeline. The initial implementation starts streams at episode boundaries and needs no burn-in approximation.

### 7.3 Efficient trainer at v4

First perform the small recurrent rollout for a chunk, then build several independent reader states, then evaluate Qwen in ordinary batches/microbatches. Qwen outputs never enter the next writer state.

The exact gradient bridge accumulates gradients at detached **leaf proxies**, then applies a final vector–Jacobian backward to the original producer outputs. This is not the same as detaching and forgetting the connection. All reader microbatches and producer backward finish before synchronization, clipping, stepping, and segment detachment [REF-AUTOGRAD].

The bridge covers **both** memory outputs and any shared trainable merged-image outputs. If those image tensors are detached without a bridge, the merger silently stops learning. v7 also includes the writer levels used by its auxiliary heads. Expose each tensor once; accumulate independent loss gradients without adding the same final level twice.

Persistently cache only verified frozen pre-merger features. Compute the trainable merger once per unique frame **within the live segment** and share its graph. Recompute merger/text outputs after optimizer changes. At inference, fixed weights permit a bounded cache of merged recent frames.

For initial cache extraction, it is acceptable to call the official visual forward under `no_grad`, retain only `last_hidden_state`, and discard the unused merged output. This computes an unnecessary merger during precomputation but avoids rewriting the encoder. It must not be confused with the differentiable training merger.

An optional v4 action-position-only output-head path gathers hidden states immediately before supervised tokens and applies the unchanged LM head only there. Prove its loss/gradient parity against full-logit SFT before claiming the saved work. This is engineering, not a new action representation.

### 7.4 Packing is a side branch

Ordinary batching has isolated batch dimensions. Flattened sequence packing is different. Qwen3.5's linear-attention/convolution state must reset at every packed example boundary, in addition to full-attention isolation [REF-QCFG, REF-QINPUT]. A block-diagonal full-attention mask alone is insufficient.

Efficient-VLN's shared-trajectory action-isolating graph is a reference for causal visibility, not a drop-in equivalent of independent readers with different memory prefixes [REF-EFF]. Do not carry another target state's labels or hidden state into a prediction. v5 can end with a documented unsupported verdict and v6/v7 still proceed.

## 8. Spatial supervision without an old-model dependency

### 8.1 Teacher/target contract

Use frozen public VGGT-1B, one independent observation per teacher sequence, in eval/no-grad mode. A batch of independent views has shape `[B,1,3,H,W]`, not `[1,B,3,H,W]`. The latter permits cross-view information flow and changes the target.

Read aggregator outputs at zero-based indices 11, 17 and 23. Remove camera/register tokens using `patch_start_idx`; do not assume every returned layer is populated. The inspected implementation may return `None` for uncached indices [REF-VGAGG]. Target width is inferred and asserted, typically 2048 from concatenated frame/global features.

Use **raw patch features** plus explicit coordinate metadata. Do not invent an additive positional-feature tensor or import an unspecified old training convention. This is a newly declared target protocol shared by v6/v7, not an exact reproduction of an earlier model.

Teacher preprocessing is deterministic aspect-preserving letterbox to 518×518. Save the scale/padding transform. Map original-image coordinates into the teacher grid before sampling. Do not resize a padded teacher grid wholesale to a rectangular Qwen grid. Target sampling and normalization are defined in the contracts.

### 8.2 v6: local geometric control

Extract only current-image Qwen hidden states after blocks 8, 16 and 24, using verified capture locations. Each independent projector is `LN(d_text) → Linear(d_text,4096) → GELU → Linear(4096,d_teacher)`. Compare with targets sampled at matching current-image spatial locations using cosine distance. Historical images, memory, text, action and padding tokens are excluded.

\[
L_{v6}=L_{nav}+\lambda_c L_{current},\qquad\lambda_c=0.3.
\]

Average per frame over valid positions and levels, then over real action states. Warm the coefficient over the first 5% of the declared target budget. No heading/progress objectives are added.

### 8.3 v7: writing and delayed recall from memory

Slots have no fixed correspondence to image patches. Do not match slot i to patch i. For level j, build a decoder with a coordinate/lag query and cross-attention to writer level H_t^(j):

\[
q=\mathrm{CoordEncode}(u)+\mathrm{LagEncode}(t-\tau),\quad
\widehat Z_{\tau,u}^{(j)}=D_j(H_t^{(j)},q).
\]

The decoder receives only memory, normalized historical image coordinates, lag, and validity masks. It cannot see the source image, teacher target, current Qwen hidden state, action, or sample identity. For writing, τ=t. For delayed retention, τ≤t−R−1 within the same episode. Initially draw eight spatial anchors from the current frame and two historical views × eight anchors per eligible state.

\[
L_{v7}=L_{nav}+0.1L_{write}+0.2L_{retain}.
\]

Average every component over valid queries/levels per action state. States with no eligible historical query contribute zero retention loss, and remain in the action-state denominator; log eligibility so the effective regularization is visible. Apply the same coefficient warmup as v6.

For the first implementation, sample retention sources within the current differentiable segment. This isolates genuine write credit at lags 5–7 when K=8,R=4. Cross-boundary queries are an explicit later sub-ablation, separately labeled. Test longer delays at evaluation.

Essential controls are geometry-free v4, current-Qwen geometry v6, write-only, retain-only, write+retain, and equal-capacity frozen-visual-feature recall. A null-memory decoder measures how much the query/teacher dataset mean can explain. All variants share initialization and target exposure.

## 9. Inference lifecycle

`PolicySession.reset(instruction)` creates fresh state, a fresh recent buffer, and cached instruction features. `act(rgb, step_id)` updates the writer once, constructs the reader prompt without target text, decodes one action, then commits the new recent buffer.

Repeated requests for the same committed step ID return the cached action only when the RGB checksum and instruction identity agree. Conflicting repeats or skipped step IDs fail. Generation of several text tokens must not trigger additional memory updates.

The action decoder uses the v0 non-thinking prefix, greedy decoding, at most 32 new tokens, and stops at the message EOS. Parse an exact allowed label after whitespace removal. On invalid output, record it and apply the declared STOP fallback; do not silently search for an action substring in free-form reasoning. Simulator scoring remains the same for every variant.

A compact export contains backbone weights, memory weights/configuration, prompt/processor protocol, and action parser settings. Auxiliary teacher/decoder weights can be retained in training checkpoints but are excluded from the deployable actor. Loading the actor in an environment without VGGT must succeed.

## 10. Evaluation and scientific interpretation

Correctness gates precede expensive training. Begin with synthetic tests, real-tokenizer golden cases, a real one-GPU update, and a tiny episode overfit. Then use a fixed scene-diverse development subset selected by complete episodes, not an arbitrary prefix of the annotation array.

Full-run metrics include SR/SPL/NE/OS, action-target throughput, actual visual/text lengths, warmed latency p50/p95/p99, peak allocated/reserved VRAM, and total preprocessing/training cost. Compare v3/v4 with the same checkpoint and batches to isolate execution changes. Compare architectures at matched action-target exposure and report compute separately.

Uniform8 maximum context is approximately L+9P; Memory64+Recent4 is L+5P+64, before delimiters. This does not imply a fivefold wall-clock speedup. The default still calls Qwen for each control action. All performance claims are measurements to obtain, not promises in this plan.

The final research question is whether retained spatial features alter memory-dependent navigation behavior beyond current-image supervision and generic recall. No additional-module count or auxiliary loss curve is sufficient evidence by itself.

## 11. Agent implementation boundaries

Every version plan states the exact existing paths to modify, new paths to create, input/output contracts, tests, and a completion gate. Do not implement later versions opportunistically. Do not install reference repositories into the main environment. Do not silently switch backbones, output heads, data mixtures, history rules or precision policies.

Missing real data metadata, simulator assets, resolved dependencies, or a source symbol that differs from the inspected revision are specific blockers to report. They are not permission to invent a value or pretend a test passed.

Read [shared contracts](09_SHARED_CONTRACTS.md) before writing code and [the reference guide](11_REFERENCE_CODE_GUIDE.md) before consulting an external implementation.
