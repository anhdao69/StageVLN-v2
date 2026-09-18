# v0 — Standalone Uniform8 SFT Implementation Plan

**Goal:** turn the uploaded plain SFT repository into an audited, reproducible starting policy with up to eight uniformly selected historical frames plus current.
**Architecture:** unchanged public Qwen3.5-4B; frozen visual backbone, trainable merger and language/output weights; one textual action per observation.
**Spec:** [master](00_MASTER_PLAN.md), [contracts C1–C4, C7, C11](09_SHARED_CONTRACTS.md).
**Important:** this is a corrected new baseline, not a numerical reproduction of an older StageVLN experiment.

## Boundaries and initial settings

R2R only; global action-state batch32; language LR1e-6, merger1e-5; current pixel bounds `576*28*28` and `16*28*28`; max sequence12800; no memory, geometry, episode BPTT, feature cache, action chunks or extra data. Use the public checkpoint, not a navigation-finetuned initialization. Preserve LICENSE and existing attribution.

v0 can retain the existing annotated image lists **only after** their uniform history rule is established. The current `max_history_frames=8` checks the list size; it does not sample history. “Nine images maximum” and “Uniform8” are different assertions.

## Existing files to modify

| File / symbol | Change |
|---|---|
| `data/data_qwen.py`: `R2RSFTDataset` | strict record schema, shared prompt builder, immutable records, per-record metadata |
| `data/data_qwen.py`: `DataCollatorForSFT` | remove silent truncation; complete-target and per-image count checks |
| `train/train_qwen.py`: `train` | shared resolved config, explicit initialization/resume, exported prompt protocol |
| `train/trainer.py`: `QwenSFTTrainer` | declared decay grouping, per-state objective, local sampler override |
| `train/sampler.py` | preserve length utilities, remove global Trainer monkeypatch |
| `train/argument.py` | explicit protocol/revision/resume controls; keep compatible entrypoint |
| `train/v0_uniform8.sh`, `setup.py`, `README.md` | independent environment, verified dependencies and clear run recipe |

All paths above are under `src/qwen_vl/` unless already prefixed `train/` or at repository root. Do not create a parallel training stack just to rename these files.

## Task 1 — Freeze the input and make launch independent

**Create:** `src/qwen_vl/config.py`, `src/qwen_vl/train/run.py`, `src/qwen_vl/tools/preflight.py`, `configs/experiments/v0_uniform8.json`, `train/run_experiment.sh`.
**Interface:** `load_run_spec(path) -> RunSpec`; `train_from_spec(spec)` delegates to the existing HF trainer.

- [ ] Compare the live repository with `inputs/current_repo/`; save the actual git revision/diff. Do not overwrite newer user changes.
- [ ] Add a launcher test that fails when no valid environment is active and no local `.venv` exists. It must not find an environment by traversing to a neighboring project.
- [ ] Validate the dependency declarations against the active environment and record exact versions, CUDA/device/kernel availability and model revision. Do not install μVLA's Transformers fork.
- [ ] Make `--resume` and `--init-checkpoint` explicit and mutually exclusive. Fresh training into a nonempty checkpoint directory fails unless an explicit resume is requested.
- [ ] Keep one configuration authority. The v0 shell wrapper resolves the v0 config and forwards whitelisted runtime overrides.

**Tests:** `tests/unit/test_config.py`, `tests/unit/test_launch_contract.py`, `tests/unit/test_resume_provenance.py`.

```python
def test_implicit_resume_is_rejected(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    # The proposed validator receives the resolved output directory and no resume.
    with pytest.raises(ValueError, match="explicit resume"):
        validate_output_directory(tmp_path, resume=None)
```

`validate_output_directory` is implemented in `config.py`. The fixture must include an output directory with ordinary logs but no checkpoint, according to the documented fresh-run overwrite policy.

## Task 2 — Audit actual records rather than invent uniformity

**Create:** `src/qwen_vl/tools/audit_dataset.py`.
**Modify:** `R2RSFTDataset` and path normalization.
**Inputs:** actual annotation JSON, image root, optional original trajectory metadata.
**Output:** `data_audit.json` plus a bounded human-readable sample report.

- [ ] Require exactly one human user message followed by one gpt assistant action for this dataset protocol. Canonicalize aliases only if explicitly supported and tested.
- [ ] Reject empty images, invalid action text, wrong final role, mismatched placeholders, paths outside the dataset root, duplicate image paths and cross-episode frames.
- [ ] Identify current observation and temporal order from verified metadata/filename conventions. Validate indices against the expected Janus-style rule; report the rule and evidence used.
- [ ] Sample and inspect states t=0,1,7,8,9 and long episodes. Test the t=100 example in C4 where available.
- [ ] If only the selected images are available and the original frame-order mapping cannot be established, report `uniformity=unverified`. The run may be called `annotated_history_sft`, not certified `uniform8`.
- [ ] Keep annotation order only after the audit passes; do not silently sort bad records or discard them without recording changed target counts.

Upstream `create_data/create_data.py::process_episode_vlnce` is a sampling/data-format reference, not proof of how the local JSON was generated [REF-JANUS].

## Task 3 — Make prompt and target identity testable

**Create:** `src/qwen_vl/data/prompting.py`, initial types in `src/qwen_vl/contracts.py`, `src/qwen_vl/tools/inspect_batch.py`.
**Modify:** `tokenize_conversation`, dataset/collator, save/export logic.

- [ ] Centralize system text, user template, image expansion, assistant non-thinking prefix and target suffix.
- [ ] Remove the dataset-only tokenizer-template mutation. Serialize the actual prompt protocol used by both training and inference.
- [ ] Reject sequence lengths above12800 before collating. Check the complete action suffix remains intact; reject both zero-target and partial-target examples.
- [ ] Derive each image's expected placeholder count from its grid/merge size. Validate each sample independently, not merely the concatenated batch total.
- [ ] Record exact image spans and target start. Build attention masks from real lengths; mask every memory-free prompt position.
- [ ] Verify saved/reloaded generation-prefix IDs equal training IDs before the first target token. Use the real tokenizer, all four actions and 1/5/9-image cases.

**Tests:** `tests/unit/test_record_schema.py`, `test_target_guard.py`, `test_collator.py`; `tests/integration/test_prompt_roundtrip.py`.

```python
def test_overlength_target_is_never_silently_truncated():
    sample = make_tokenized_fixture(length=33, target_start=30, image_count=1)
    with pytest.raises(ValueError, match="Overlength"):
        collator_with_limit(32)([sample])
```

Implement those fixture helpers in `tests/conftest.py`; one variant must preserve all image tokens while cutting an action, reproducing the actual current defect.

## Task 4 — Verify optimization and sample weighting

**Modify:** `QwenSFTTrainer.create_optimizer`, add a local `_get_train_sampler` override and `compute_loss` only as required by C7.
**Create:** `src/qwen_vl/train/losses.py` with `navigation_loss_per_state(logits, labels) -> Tensor[B]`.

- [ ] Keep the visual backbone frozen and its merger trainable; verify parameter identity coverage and tied output/input weights.
- [ ] Use a declared normalization/bias-excluded decay rule. Validate custom Qwen RMSNorm, bias, merger and regular linear weights explicitly.
- [ ] Verify per-state token means with unequal label lengths and reader batch sizes. Do not double-shift labels or double-divide accumulation losses.
- [ ] Check the same synthetic effective batch under HF accumulation equals a direct global objective; pin the actual Trainer behavior before a full run.
- [ ] Remove the global `Trainer._get_train_sampler = ...` mutation. Keep the existing length grouping only inside this IID trainer.

One useful primitive in `losses.py` is `cross_entropy(logits[:,:-1].float(), labels[:,1:], reduction='none')`, reshaped and masked per row. Raise if any row has no target. The Trainer's update-level scaling still needs the explicit test; the primitive alone does not establish distributed correctness.

## Task 5 — Add a minimal actor/evaluation contract

**Create:** `src/qwen_vl/eval/action_decoder.py`, `session.py`, `runner.py`, `metrics.py`, `latency.py`.

Implement the memory-free `PolicySession` with audited Uniform8 history selection and the same prompt builder. Fresh Qwen cache per navigation decision; cache only within generation. Greedy one-action decoding and invalid-output handling follow master §9.

The simulator adapter receives the real environment's observation/action mapping, stop behavior and episode specification. They are not present in the attachment. Do not invent a nominal simulator turn/movement setting. Until the assets are supplied, use an explicit fake-environment test and mark SR/SPL evaluation unrun.

## Commands after implementation

```bash
python -m pytest tests/unit -q
python -m qwen_vl.tools.preflight --config configs/experiments/v0_uniform8.json
python -m qwen_vl.tools.audit_dataset --config configs/experiments/v0_uniform8.json
python -m qwen_vl.tools.inspect_batch --config configs/experiments/v0_uniform8.json --count 8
MAX_STEPS=2 MAX_SAMPLES=32 SAVE_FINAL_MODEL=False bash train/v0_uniform8.sh
```

The source snapshot's initial one-step warmup can make a first-step weight-change check misleading. Verify a nonzero-LR update, finite gradient norms, a changed trainable weight and an unchanged frozen vision weight. The smoke subset is for mechanics, not navigation performance.

## Completion gate

Pass schema/target/prompt/optimizer tests; prove or explicitly withhold Uniform8 certification; complete a real Qwen save/reload/update smoke; provide resolved config/provenance. A trained baseline used for later model comparisons also needs closed-loop evaluation under the fixed simulator protocol. Do not start v1 by changing the target budget or checkpoint provenance.
