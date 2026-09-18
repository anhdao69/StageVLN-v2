# Codebase Map — Existing Evidence and Proposed Additions

This map refers to the **13 files actually present in the uploaded bundle**. It is not a claim that the user's live git checkout has no other files. Compare the current checkout before applying edits. The extracted input snapshot is at [inputs/current_repo](inputs/README.md).

## Existing files: preserve, repair, then extend

| Existing path | Observed role | Planned ownership/change |
|---|---|---|
| `README.md` | plain Qwen R2R training instructions; inherited environment reference | v0 standalone setup, exact recipe, evidence links; remove unverified performance promises |
| `LICENSE` | supplied repository license/attribution | preserve; do not replace with a new project's invented copyright |
| `setup.py` | Python3.12 and dependency declarations | v0 inspect/lock actual environment, declare directly used dependencies; do not install reference stacks |
| `configs/datasets/newton_r2r_uniform8.json` | R2R annotation/root paths | keep machine configuration; expand env syntax; actual paths remain user-provided |
| `src/qwen_vl/__init__.py` | package marker | keep namespace |
| `src/qwen_vl/data/__init__.py` | package marker | export data interfaces as needed |
| `src/qwen_vl/data/data_qwen.py` | JSON loading, image processing, prompt tokens, collator | v0 schema/prompt/target fixes; v1 history from canonical episodes; delegate new features to small modules |
| `src/qwen_vl/train/argument.py` | HF model/data/training arguments | preserve backward-compatible fields; add resolved RunSpec conversion once |
| `src/qwen_vl/train/sampler.py` | length grouping and global Trainer monkeypatch | preserve pure utilities; remove global monkeypatch; never use for recurrent streams |
| `src/qwen_vl/train/train_qwen.py` | plain Qwen loading/freezing/training/save | v0 safe entrypoint and export; retain IID HF path |
| `src/qwen_vl/train/trainer.py` | custom merger LR groups | v0 exact parameter groups/normalization; share group-construction helper with temporal backend |
| `train/v0_uniform8.sh` | torchrun/ZeRO2 launcher | thin standalone config wrapper with explicit environment/resume |
| `train/zero2.json` | existing HF baseline engine config | keep for v0/v1; not automatically reused with manual temporal gradient bridge |

Do not implement a second image processor, an unrelated OpenVLA wrapper or a copy of an entire navigation repository. The old StageVLN checkout is not an implicit dependency.

## Proposed modules grouped by introduction version

Paths in this table are **new files to create**, unless an existing path above is mentioned. Small adjacent helpers may be combined if their interfaces/tests remain clear; do not scatter one function across many modules.

| Version | Proposed paths under `src/qwen_vl/` | Responsibility |
|---|---|---|
| v0 | `config.py`, `contracts.py`, `train/run.py` | strict run definition, shared records, dispatch to existing trainer |
| v0 | `data/prompting.py`, `train/losses.py` | common train/inference prompt and invariant per-state action loss |
| v0 | `tools/preflight.py`, `audit_dataset.py`, `inspect_batch.py` | real environment/data/token evidence |
| v0 | `eval/action_decoder.py`, `session.py`, `runner.py`, `metrics.py`, `latency.py` | one-action actor and simulator boundary, no invented simulator assets |
| v1 | `data/history.py`, `episode_manifest.py`, `tools/build_episode_manifest.py` | exact selectors and complete canonical episodes |
| v2 | `models/qwen_adapter.py`, `navigation_policy.py` | official visual outputs, explicit embeddings/positions and ownership |
| v2 | `models/memory_writer.py`, `instruction_encoder.py`, `memory_adapter.py` | pure small updater, text conditioning and reader prefix |
| v2 | `data/episode_stream.py`, `train/episode_trainer.py` | chronological streams and BPTT segments |
| v2 | `train/distributed_grad.py`, `checkpointing.py` | one synchronization point, coherent model/stream resume |
| v3 | `eval/memory_probes.py` | memory interventions and diagnostic reports |
| v4 | `data/feature_store.py`, `tools/precompute_features.py` | audited frozen feature storage/reuse |
| v4 | `train/gradient_bridge.py`, `tools/benchmark.py` | exact producer/reader gradient boundary and timing |
| v5 | `tools/inspect_packing_support.py`, optional `models/packed_reader.py` | feasibility report first; implementation only if supported |
| v6 | `spatial/teacher.py`, `targets.py`, `current_alignment.py` | isolated teacher targets and current-image loss |
| v7 | `spatial/memory_decoder.py`, `query_sampler.py` | memory-only geometric queries and delayed supervision |

`tools/audit_dataset.py` etc in abbreviated rows are inside `src/qwen_vl/tools/`. New package directories receive `__init__.py` files. These do not imply public API stability beyond the contracts.

## Configuration layout

Add `configs/experiments/` JSON files for the eight version conditions and named controls. Dataset location stays in `configs/datasets/`. The RunSpec combines them once and saves the resolved result. Avoid parallel YAML/JSON/HF argument sources that disagree about batch size or history.

The user-supplied cluster paths are not portable defaults for all machines. Keep them in the machine dataset config; require explicit overrides where nonexistent. The standalone launcher uses an active compatible environment or its own `.venv`, not the old neighboring environment.

## Test layout and responsibility

`tests/unit/` runs without model weights or simulator data: selectors, schema, target guards, tensor permutations, state resets, loss normalization, bridge and coordinate algebra.

`tests/integration/` has explicit markers for real tokenizer/model/teacher/distributed tests. Missing assets produce an informative skip in a developer run but **do not count as passing acceptance evidence** for a full model milestone.

Simulator tests require the actual evaluation split and environment configuration. A fake simulator tests the interface only. Metrics, action semantics, episode split and stopping threshold are recorded from the real setup, not guessed from another paper.

## Dependency direction

```text
config/contracts
  ├── data + prompt construction
  ├── actor/model interfaces
  └── trainers/evaluation

Qwen backbone + small writer → actor
VGGT target adapter + auxiliary decoders → training only
```

`eval/session.py` never imports the geometry package merely to load an actor. `memory_writer.py` never imports the Qwen trainer or simulator. `feature_store.py` never stores a trainable recurrent state as an immutable feature.

## Missing inputs that an agent must not invent

Actual local annotations/images, complete episode metadata, a resolved public model snapshot, the working PyTorch/CUDA/kernel environment, and the navigation simulator/evaluator protocol are not included in the attachment. StageVLN source code is **not** on this missing-input list: the standalone plans do not require it.
