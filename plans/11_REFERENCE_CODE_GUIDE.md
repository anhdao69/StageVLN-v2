# Verified Reference-Code Guide for the Coding Agent

**Inspected:** 2026-09-18. **Purpose:** make outside references concrete without importing an unrelated robotics stack.

The project source of truth is the uploaded Qwen repository plus the new master/contracts. External code provides specific patterns and API evidence. The new gated writer, memory read interface and spatial-query objectives are **project design choices**, not a claimed verbatim μVLA implementation.

## 1. Minimal code to provide at each stage

| Stage | Read these references | Do not provide/install as a dependency |
|---|---|---|
| v0–v1 | installed Transformers5.3.0 Qwen files; public Qwen config/template; Janus data builder | an entire StageVLN model or checkpoint |
| v2–v3 | Qwen interfaces; μVLA memory/reset/TBPTT/episode files; VPWEM Q-Former | OpenVLA/μVLA model, manipulation action head or custom Transformers fork |
| v4 | Qwen pre-merger output; PyTorch autograd docs; VPWEM feature-pipeline example | detached cache code copied without gradient review |
| v5 optional | Efficient-VLN revised paper; actual installed Qwen layer/kernel source | a guessed “Efficient-VLN compatible” repository |
| v6–v7 | official VGGT model/aggregator; this package's target/query contracts | old StageVLN training helpers or runtime geometry fusion |

## 2. Qwen — the first implementation reference

Repository: https://github.com/huggingface/transformers/tree/v5.3.0

- **[REF-QVISION]** `src/transformers/models/qwen3_5/modeling_qwen3_5.py`: `Qwen3_5VisionModel.forward`, `rot_pos_emb`, `fast_pos_embed_interpolate`, merger class. Read the returned `last_hidden_state`/`pooler_output` distinction and feature order. These support the external writer without rewriting the visual backbone.
- **[REF-QINPUT]** same file: `Qwen3_5Model.get_rope_index`, `compute_3d_position_ids`, `forward`, `Qwen3_5ForConditionalGeneration.forward`, dynamic cache and gated-delta layer. Preserve the exactly-one IDs/embeddings contract and mixed attention-state semantics.
- `src/transformers/models/qwen3_vl/processing_qwen3_vl.py`: image placeholder expansion and multimodal metadata.
- `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`: independent-image preprocessing and grid/merge conventions.
- `src/transformers/loss/loss_utils.py`: `ForCausalLMLoss`; verify label shifting and reduction rather than duplicating a shift.

Public checkpoint metadata **[REF-QCFG]**:
- https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/preprocessor_config.json
- https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/chat_template.jinja

These are the version5.3.0 APIs inspected, not a promise that another installed release has identical behavior. Do not copy generated model source into the project just to add a prefix; first implement the wrapper and parity tests.

## 3. μVLA — useful lifecycle/training code, different recurrence location

Project: https://avanturist322.github.io/mu-vla/
Paper: https://arxiv.org/abs/2606.12497
Repository: https://github.com/CognitiveAISystems/muVLA

| Reference | Exact path / focus | Use in this project |
|---|---|---|
| **[REF-MUMEM]** | `prismatic/models/memory.py`: `MemoryModule`, learned initial memory, `get_initial_state`, episode reset, EMA branch | understand initialization/reset boundaries; implement our own pure explicit state interface |
| **[REF-MUTRAIN]** | `vla-scripts/finetune.py`: `run_forward_pass`, memory return, TBPTT/accumulation/reset handling | trace temporal gradient lifetime and optimizer ordering; do not port training hyperparameters blindly |
| **[REF-MUDATA]** | `prismatic/vla/datasets/libero_episodic_dataset.py`: episodic iterable and first/last information | learn episode-slot metadata pattern, not LIBERO transforms or RLDS labels |
| contextual | `prismatic/extern/hf/modeling_prismatic.py` | locate how backbone output produces memory; understand why it is not our lightweight writer |
| diagnostics | `prismatic/models/memory_diagnostics.py` | examples of state diagnostics; implement only those useful for this project |
| environment | `SETUP.md`, `pyproject.toml` | inspect only to understand incompatible stack assumptions |

**Critical difference:** μVLA's small `MemoryModule` is not a complete standalone transformer updater. Its integration extracts updated memory from the large backbone. Copying only that file will not implement `f_phi` in our master plan. Our writer is explicitly defined in master §6.3 and must be implemented as a separate small module.

The inspected μVLA setup uses Python3.10 and a memory-aware custom Transformers fork. Its docs mention fork revision `9dbc09f5`; this is a short revision printed upstream, **not a verified lock for our project**. Do not install that fork into this repository's Qwen5.3.0 environment. Read the EMA/detach branches critically: the main gradient path here uses declared TBPTT, not unconditionally detached updates.

## 4. VPWEM — structural reference for a small compressor

Repository: https://github.com/HarryLui98/code_vpwem
Paper: https://arxiv.org/abs/2603.04910

- **[REF-VPQ]** `cleandiffuser/nn_memory/qformer_memory.py`: `QFormerLayer`, `QFormerMemory`, query/condition caches and reset behavior. Borrow attention-block organization, not the internal detached-cache policy.
- **[REF-VPPIPE]** `pipelines/longmemdpptp_robomimic_image_emb.py`: precomputed visual-embedding dataflow. Borrow the distinction between visual feature storage and downstream trainable processing; do not port the diffusion/continuous-action objective.

Our writer updates on every current observation and learns through K-step BPTT. Its state may overlap the recent window. It is not an exact reproduction of VPWEM's out-of-window compressor, cache size, update stride or no-BPTT implementation.

## 5. JanusVLN — data conventions, not the actor

Repository: https://github.com/MIV-XJTU/JanusVLN
**[REF-JANUS]** `create_data/create_data.py::process_episode_vlnce`.

Inspect the full episode image ordering, Uniform8 index selection, generated record IDs, action extraction and instruction template. Do not assume the local annotations match upstream merely because they use the same format. Do not port geometry fusion, caches, other dataset mixtures or ScaleVLN augmentation into v0.

## 6. VGGT — teacher tensors and preprocessing boundary

Repository: https://github.com/facebookresearch/vggt
Public checkpoint: https://huggingface.co/facebook/VGGT-1B

- **[REF-VGAGG]** `vggt/models/aggregator.py`: `Aggregator.forward`, `cached_layer_indices`, `patch_start_idx`, frame/global attention and output concatenation.
- **[REF-VGMODEL]** `vggt/models/vggt.py`: `VGGT` construction and aggregator access. The prediction heads need not run to obtain features.

Read the input `[B,S,3,H,W]` distinction carefully. This plan uses S=1, with independent images on B. The letterbox coordinate mapping and raw-feature target convention are fully specified in C9. There is no need to find an old StageVLN function that supposedly implements them.

## 7. Other primary references

- **[REF-AUTOGRAD]** https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html — explicit output gradients used by the bridge. Record installed PyTorch version; “stable” URLs may move.
- https://docs.pytorch.org/docs/stable/checkpoint.html — use non-reentrant activation checkpointing where required, with parity tests.
- https://docs.pytorch.org/docs/stable/distributed.html — collective synchronization semantics.
- **[REF-EFF]** https://arxiv.org/html/2512.10310v2 — revised Efficient-VLN paper, especially action-isolating packed training. This is a paper-only reference here; no code repository is claimed verified.

The StageVLN paper is an **optional conceptual reading** at v6 about training-only current-image spatial alignment. All required implementation information is restated in these plans. Earlier project code or checkpoints are not required.

## 8. Obtain a compact offline reference pack

The optional collector uses the verified repositories/files listed in `references.json`:

```bash
python tools/fetch_references.py --manifest references.json --output reference_code
```

Run this from this planning-package directory on a machine with Git and network access. It resolves each branch/tag once, saves a full commit hash and file hashes to `reference_code/references.lock.json`, and exports only requested source/license files. It installs nothing and downloads no model weights or datasets. It never executes repository code.

Give the coding agent `reference_code/`, this guide, the master, contracts and one version plan. Full repository checkouts are optional when an import needs inspection; use the recorded commit, not a later moving branch. Keep reference trees outside the trainable package and do not add them to PYTHONPATH.

### Verification limitations

The web-visible paths/symbols above were inspected while drafting. Most repositories were inspected at mutable `main`; full commit hashes were not successfully established in this environment. They are deliberately not fabricated. The collector resolves and locks them when run. If a required path moved or a symbol differs, report it and reconcile the adapter against that exact revision before coding.

The collector itself was syntax/local-fixture tested only; external repository fetching was not executed here. External source files are therefore **not already bundled** in this archive. The user's input code snapshot is bundled separately and clearly labeled unmodified.
