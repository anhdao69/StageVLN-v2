# Standalone Qwen Navigation Memory — StageVLN-v2 planning package

**Revision:** 2026-09-18 · standalone, code-grounded replacement for the earlier plans.
**Deliverable status:** implementation specifications, not a trained model or an applied repository patch.

## What “from scratch” means here

Build a new navigation/memory project from the uploaded **plain Qwen3.5 R2R SFT repository**. Initialize the backbone from the public pretrained `Qwen/Qwen3.5-4B` checkpoint, not a StageVLN or other navigation-finetuned checkpoint. Initialize new memory/auxiliary modules randomly. This does **not** mean pretraining a four-billion-parameter VLM from random weights.

There is no dependency on an earlier StageVLN checkout, environment, hidden helper, or checkpoint. JanusVLN is a data-format/sampling reference. μVLA and VPWEM are read-only architectural/training references. VGGT becomes a training-only dependency at v6. Everything through spatial-memory supervision is specified here; no subsequent policy-optimization stage is included.

## Give a coding agent these files first

1. [Master design](00_MASTER_PLAN.md).
2. [Shared contracts](09_SHARED_CONTRACTS.md).
3. [Actual codebase map](10_CODEBASE_MAP.md).
4. The **one version plan** you are asking it to implement.
5. [Agent handoff prompt](13_AGENT_HANDOFF.md), with the version selected.

The [input snapshot](inputs/README.md) is included so another agent can understand the starting point without the previous conversation. Run commands in version plans refer to files that the tasks will create; they are not claims that those commands already exist in the supplied repository.

## Version ladder

| Version | Model / change | Plan |
|---|---|---|
| v0 | Audited Uniform8 + current; standalone plain SFT | [v0](01_v0_uniform8.md) |
| v1 | Recent4 + current; Current-only control | [v1](02_v1_sw4_current_control.md) |
| v2 | External Memory64 + current; sequential reference trainer | [v2](03_v2_memory_only.md) |
| v3 | External Memory64 + Recent4 + current | [v3](04_v3_memory_sw4.md) |
| v4 | Same policy; feature reuse, batched readers, exact microbatch gradients | [v4](05_v4_efficient_training.md) |
| v5 | Optional sequence-packing feasibility branch | [v5](06_v5_optional_packing.md) |
| v6 | Same policy + current-image geometric distillation | [v6](07_v6_spatial_current.md) |
| v7 | Same policy + geometric writing/retention supervision of memory | [v7](08_v7_spatial_memory.md) |

```text
v0 → v1 → v2 → v3 → v4 ──→ v6 → v7
                         └──→ v5 (optional)
```

Version order is software development order, **not** a requirement to initialize one experiment from the previous version's trained weights. v6 and v7 use v4 infrastructure but are independent experimental conditions.

## Additional documents

- [Reference-code guide](11_REFERENCE_CODE_GUIDE.md): exact verified files/symbols, what to borrow, what not to port, and how to collect code for an offline agent.
- [Experiment protocol](12_EXPERIMENT_PROTOCOL.md): initialization, data budgets, correctness gates, and quality/cost measurements.
- [Revision changes](14_CHANGELOG.md): important differences from the previous plans.
- [Package validation](VALIDATION.md): what was checked while creating this package and what still requires the real repository/hardware.

`tools/fetch_references.py` is an optional **read-only source collector**. It fetches no model weights or datasets and installs nothing. Repository branches are resolved to commits on first collection and recorded in a lock file; the guide does not fabricate unverified commit hashes. Internet access is required to run this collector. External source trees are not already bundled here. `tools/validate_plan_algebra.py` contains the small CPU checks described in the validation report; it is not a navigation implementation.

## Authority and unresolved evidence

The supplied code is the evidence for existing paths and behavior. The master/contracts are the authority for **new design decisions**. Outside repositories are references, not drop-in implementations. If the live repository differs, report the diff before changing behavior.

Actual annotations/images, a Qwen checkpoint, a deployed environment, and simulator configuration were not available for this revision. Their audit is an explicit v0/v1 gate, not a fact silently assumed by these plans.
