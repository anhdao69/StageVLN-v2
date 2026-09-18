# v1 — Recent4 and Current-only Implementation Plan

**Goal:** isolate explicit history before introducing memory.
**Architecture:** same Qwen model and action likelihood as v0. `v1_sw4` uses previous four observations plus current; `v1_current` uses current only.
**Spec:** [master](00_MASTER_PLAN.md), [contracts C2–C4](09_SHARED_CONTRACTS.md).
**Prerequisite:** v0 prompt, target, initialization and evaluation gates. No geometry or recurrence.

## Changes allowed

Only historical observation selection changes in the primary v0/v1 comparison. Keep native image resolution/tokenization, training weights, objective, target count and action interface fixed. Generate fresh runs from the same public model revision. Do not warm-start the main v1 comparison from a trained Uniform8 policy.

## Task 1 — Build a complete canonical episode manifest

**Create:** `src/qwen_vl/data/episode_manifest.py`, `src/qwen_vl/tools/build_episode_manifest.py`.
**Modify:** `contracts.py` to add `FrameKey`, `FrameRecord`, `EpisodeRecord`.
**Interface:** `build_episode_manifest(annotation_path, dataset_root, metadata_path=None) -> ManifestReport`; persisted manifest is JSONL plus a fingerprint/index file.

- [ ] Recover episode identity, instruction identity and physical observation order from verified metadata.
- [ ] Use every step record's last/current image as one possible source of the full episode. Deduplicate only exact duplicate observations/labels under the same instruction.
- [ ] Verify expected observation coverage using original metadata or the audited full image list. Detect missing interior observations; sorting does not fix gaps.
- [ ] Check that the instruction is constant within a stream. Different instructions for the same physical trajectory create different episode keys.
- [ ] Preserve terminal semantics. Never invent an action for an unlabeled terminal view.
- [ ] Store original record references so a new state can be traced back to the annotated source.

If only nine sampled images are known for an entire trajectory, Recent4 and chronological recurrence cannot be constructed correctly. Treat that as a concrete missing-data blocker, not as permission to reinterpret sparse samples as consecutive observations.

## Task 2 — Make selection a shared pure function

**Create:** `src/qwen_vl/data/history.py`.
**Interface:** `history_indices(t, mode, recent=4)`, defined in C4.
**Modify:** dataset state construction and the evaluator's frame selection to call the same function.

```python
def test_recent_is_not_uniform_tail():
    assert history_indices(100, "recent", 4) == [96,97,98,99,100]
    assert history_indices(100, "uniform8")[-5:] != history_indices(100, "recent", 4)

def test_current_and_early_windows():
    assert history_indices(0, "recent", 4) == [0]
    assert history_indices(2, "recent", 4) == [0,1,2]
    assert history_indices(20, "recent", 0) == [20]
```

- [ ] Test all t from0 to512 for uniqueness, increasing order, no future frames and current at the end.
- [ ] Ensure exactly R preceding frames at sufficiently late states, not R including current.
- [ ] Build prompts from manifest observations and the verified instruction renderer; do not delete image markers with an unanchored string replacement that also changes the instruction.

## Task 3 — Integrate without refactoring the backbone

**Modify:** `data/data_qwen.py`, `train/argument.py`, `config.py`.
**Create:** `configs/experiments/v1_sw4.json`, `v1_current.json`; optional thin shell wrappers.

Keep the HF/ZeRO2 IID path. Dataset length and action-label distribution must equal the audited v0 set when all records are usable. On states t≤4, Recent4 includes all observations; compare prompts with the corresponding v0 states where the image lists match.

Export the selected history mode with the model. The evaluator must reject a checkpoint declared Recent4 when an incompatible override asks for Uniform8 unless explicitly running a labeled cross-protocol diagnostic.

## Task 4 — Freeze the experiment controls

**Create:** a split manifest and experiment table; reserve names `v1_current_ep` and `v1_sw4_ep` for the future temporal-backend controls at v2/v3.

- [ ] Same public checkpoint revision, seed, target budget, pixel limits, action objective and evaluator across v0/v1.
- [ ] No-memory flags allocate no memory parameters and insert no memory delimiters/placeholders.
- [ ] Report actual visual-token counts and latency rather than assuming a factor-of-two gain.
- [ ] Include tiny-subset overfit and full development-subset closed-loop evaluation.

## Tests and commands after implementation

```bash
python -m pytest tests/unit/test_history.py tests/unit/test_episode_manifest.py -q
python -m qwen_vl.tools.build_episode_manifest --config configs/experiments/v1_sw4.json
python -m qwen_vl.tools.inspect_batch --config configs/experiments/v1_sw4.json --count 8
CONFIG=configs/experiments/v1_sw4.json MAX_STEPS=2 bash train/run_experiment.sh
CONFIG=configs/experiments/v1_current.json MAX_STEPS=2 bash train/run_experiment.sh
```

The script resolves `CONFIG` to the documented `--config` interface. Implement it once, not as duplicated training arguments in each wrapper.

## Completion gate

Manifest completeness is established; selector and online/offline parity tests pass; both no-memory configurations train/evaluate; the data/target budget is unchanged. Weak Current-only performance is not a code failure. It establishes the comparator for v2.

**External code needed:** JanusVLN data builder only for source conventions [REF-JANUS]. μVLA is not needed at this stage.
