# Coding-Agent Handoff

Use this file together with the master, shared contracts, codebase map and **one** version plan. The existing repository is the uploaded plain Qwen SFT code, not an older StageVLN checkout.

## Copy-ready first task

```text
Implement v0_uniform8 only, using this planning package.

Read, in order:
1. 00_MASTER_PLAN.md
2. 09_SHARED_CONTRACTS.md
3. 10_CODEBASE_MAP.md
4. 01_v0_uniform8.md
5. inputs/README.md and the actual repository files listed there.

This is a standalone navigation project initialized from public pretrained
Qwen/Qwen3.5-4B. Do not load an earlier StageVLN/navigation checkpoint, use a
neighboring project's environment, import its helpers, or assume its tests passed.
Do not add memory, geometry, action chunks, new datasets or packing in v0.

First compare the live repository against the supplied input snapshot. Preserve
newer user edits. Report the concrete paths/functions that differ. Preserve the
existing namespace qwen_vl and license/attributions.

Implement the version's tasks in small tested changes. Add failing tests for the
specified current defects before fixing them. Separate synthetic tests from real
Qwen/data/GPU/simulator tests. Use the master/contracts for new behavior and the
uploaded code for evidence of existing behavior.

If the real data cannot establish Uniform8 sampling, report that exact missing
metadata and do not label an unverified history list as certified Uniform8.
Never silently truncate action targets, assume a saved chat template matches
training, or resume from a checkpoint merely because one exists in the output dir.

Do not launch full-dataset training or edit later versions as a side effect.
Run only the requested tests and bounded smoke unless explicitly authorized.

Finish with:
- exact files changed and why;
- commands executed and their actual outcomes;
- tests skipped/unavailable and required assets;
- resolved initialization/data/history/optimization settings;
- a concise acceptance report for this version;
- the precise next blocking issue, if any.
Do not claim navigation quality or production readiness from a smoke test.
```

## Prompt for a later version

```text
Implement the selected version plan only, on the now-current repository.
Use 00_MASTER_PLAN.md, 09_SHARED_CONTRACTS.md, 10_CODEBASE_MAP.md and the selected
version's Markdown file as the specification. Confirm the preceding required gates
using actual artifacts/tests; version numbers in filenames are not proof.

Keep the deployment input/action interface and checkpoint/data budget unchanged
unless this version explicitly changes them. Implementation order does not mean
warm-starting each main experiment from the previous trained model.

Consult only the external files mapped to this version in
11_REFERENCE_CODE_GUIDE.md. Read-only reference code may be supplied under
reference_code/. Do not install μVLA/OpenVLA dependencies or copy its backbone
recurrence into our lightweight external writer. Do not detach producer tensors
unless the specified gradient bridge restores their gradients.

Report missing source/API differences instead of inventing compatibility. Do not
start full training or implement a later version without explicit authorization.
Produce the same evidence-based completion report as the v0 handoff.
```

Select exactly one:

| Requested milestone | Plan file | Required previous software |
|---|---|---|
| Uniform8 baseline | `01_v0_uniform8.md` | uploaded repository |
| Sliding/current controls | `02_v1_sw4_current_control.md` | v0 |
| Memory-only | `03_v2_memory_only.md` | v1 and complete episodes |
| Memory + sliding | `04_v3_memory_sw4.md` | v2 |
| Efficient readers | `05_v4_efficient_training.md` | v3 |
| Optional packing | `06_v5_optional_packing.md` | v4 |
| Current geometry | `07_v6_spatial_current.md` | v4; v5 not needed |
| Memory geometry | `08_v7_spatial_memory.md` | v4 and v6 teacher adapter, not v6 trained weights |

## What to give an offline coding agent

Provide this package, the actual repository, and the relevant files collected with `tools/fetch_references.py`. The input snapshot is a readable baseline but is not a git checkout or repaired implementation. Supply actual annotation samples/full manifest and environment metadata when they are needed. Teacher/model weights and simulator assets are not embedded here.

The reference collector is optional; an agent with the actual installed Transformers source and verified repository checkouts can use those instead. Record exact revisions. No external service/plugin or previous conversation memory is required to understand the plan.
