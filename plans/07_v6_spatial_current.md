# v6 — Current-Image Geometric Distillation Implementation Plan

**Goal:** establish the local spatial-supervision control on the new standalone memory policy.
**Architecture:** the v4 deployed actor is unchanged. A frozen VGGT teacher supplies training targets to projectors attached to current-image Qwen hidden states.
**Spec:** [master §8](00_MASTER_PLAN.md), [contracts C9–C11](09_SHARED_CONTRACTS.md).
**Initialization:** same public Qwen checkpoint as other main conditions. Do not initialize the main v6 experiment from an extra-trained v4 model without declaring the extra budget.

## Relation to the older paper

Current-image multi-level geometric distillation is conceptually related to the supplied StageVLN paper. Its code/checkpoint is **not required**. This plan defines its own raw-feature target, coordinates, tensor shapes, loss and implementation. It does not copy heading/progress objectives or claim numerical reproduction of an older geometric recipe.

## Task 1 — Add the isolated training-only teacher

**Create:** `src/qwen_vl/spatial/teacher.py`, `targets.py`.
**Reference:** official `vggt/models/vggt.py` and `vggt/models/aggregator.py` [REF-VGMODEL, REF-VGAGG].
**Interface:** `GeometryTeacher.encode(images, frame_keys) -> GeometryTargets`, containing selected patch grids, valid masks and recorded image transforms.

- [ ] Load public `facebook/VGGT-1B` from a recorded immutable revision. Freeze all parameters, eval mode, no_grad forward.
- [ ] Call the aggregator directly; depth/pose/track output heads are not needed.
- [ ] Independent images are `[B,1,3,518,518]`. Reject accidental `[1,B,...]` multiview treatment in this protocol.
- [ ] Read layer indices11,17,23; verify returned entries are populated. Remove special tokens using returned `patch_start_idx` and verify patch count/width.
- [ ] Keep raw concatenated patch features. Do not add an imaginary teacher positional tensor.
- [ ] Actor imports remain independent of VGGT. Load the teacher lazily only for spatial training or target generation.

## Task 2 — Make image-coordinate alignment exact

Teacher images use the deterministic letterbox transformation in C9. Qwen retains its native rectangular image preprocessing. Save actual rounded resize dimensions and integer padding for every teacher input.

**Tests:** `tests/unit/test_geometry_coordinates.py`, `test_teacher_shapes.py`.

Construct a synthetic teacher grid whose channels encode x/y coordinates. Sample known original-image center/edge coordinates and assert the expected mapped locations with `align_corners=False`. Include portrait and landscape images. A direct padded-grid resize that stretches padding into the student image must fail this test.

Test independent-view teacher batching by comparing one image alone with the same image in a batch. Use actual model smoke tests when weights are available; synthetic shape tests do not prove the teacher's output interface.

## Task 3 — Capture only current-image student states

**Create:** `src/qwen_vl/spatial/current_alignment.py`.
**Modify:** `models/qwen_adapter.py` to optionally capture selected layer outputs, maintaining v4 parity when capture is disabled.

Capture states after decoder blocks8,16,24. Verify whether the installed model's hidden-state tuple includes the input embedding state and document the correct indices. Forward hooks belong on decoder outputs, not arbitrary attention sublayers. Under activation checkpointing, avoid retaining duplicate recomputation outputs or detached states.

Select the **current** `ImageSpan` for each example. Exclude historical images, memory placeholders, language, actions and padding. Use independently derived native merged-grid centers to sample teacher targets.

Project each level using `LN(d_text) → Linear(d_text,4096) → GELU → Linear(4096,d_teacher)`. Dimensions are checked against loaded configurations, not only hardcoded defaults. There are three independent projectors.

## Task 4 — Add the explicitly normalized loss

For each state, average epsilon-stabilized cosine distance over valid current-image positions and three levels, then average over all real action states through C7. The objective is:

\[
L=L_{nav}+\lambda_c L_{current},\quad \lambda_c^{max}=0.3.
\]

Warm lambda from0 to0.3 over the first5% of the fixed supervised-action-state budget. The schedule uses a saved target counter, not reader microbatch count. Keep nav supervision unchanged.

- [ ] Teacher parameters never receive gradients.
- [ ] Student projectors and selected Qwen layers receive auxiliary gradients.
- [ ] Action-suffix perturbation leaves captured current-image states unchanged under causal visibility.
- [ ] Setting lambda0 reproduces the v4 path and gradients.
- [ ] Different reader microbatch partitions preserve the combined objective.

## Task 5 — Add a practical target-storage mode

Start the small pilot with on-the-fly teacher targets and a bounded in-process cache keyed by immutable frame/teacher identity. This establishes numerical correctness without a multi-terabyte preprocessing commitment.

Dense three-level teacher features can be large. Before a full cache, measure bytes/frame and multiply by the actual unique-frame count. Record any compression/storage dtype and compare its target/loss error. v6 may query dense grids at current-image positions; v7 may use sparse anchors. A sparse matched-current control helps separate target density from supervision placement.

Do not report a warm-cache training speed without preprocessing cost or imply that all historical targets fit in GPU memory.

## Task 6 — Export and experiment controls

**Create:** `configs/experiments/v6_spatial_current.json`.
**Modify:** checkpoint/export code to distinguish training modules from actor modules.

The geometry-free v4 and v6 main experiments have identical starting checkpoint, target budget, history/memory settings and evaluator. Only auxiliary training differs. Export an actor that loads and acts when VGGT is not installed. Auxiliary projectors are not required at deployment.

## Completion gate

Coordinate/order tests pass; teacher is frozen and single-view; current token selection is correct; lambda0 parity holds; a real teacher+Qwen update succeeds; deployable actor excludes the teacher. Record SR/SPL/NE and added teacher training cost. A positive geometry-probe loss trend is not a substitute for navigation results.
