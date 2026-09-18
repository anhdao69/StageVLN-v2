# Package Validation and Limits

## What this package is

A revised master specification and implementation plans grounded in the uploaded plain Qwen SFT snapshot. It is **not** a patched repository, trained model, or certification that the real dataset/GPU stack is ready.

## Checks performed while preparing the package

The included `tools/validate_plan_algebra.py` was executed on CPU with PyTorch2.10.0+cpu. All **nine** checks passed:

| Check | What it establishes |
|---|---|
| History selection | Exact Uniform8/Recent4/Current-only examples and causal bounds across513 timesteps |
| Feature ordering | Rectangular merge-block/raster permutation round-trips |
| Reset/detach | Selective resets preserve gradients to initial slots; detach preserves values but cuts the graph |
| Loss partitioning | Per-state weighting is invariant to uneven microbatches and unequal target-token lengths |
| Gradient bridge | Direct and bridged gradients agree on an8-step toy with shared merger outputs and ancestor/descendant auxiliary losses |
| Temporal causality/credit | Future-input changes do not alter earlier states; a late loss reaches an early input in the live graph |
| Delay eligibility | K8/R4 within-segment off-window delays are5–7 |
| Global normalization | Algebra with unequal/empty ranks matches a union-of-states mean; this is not a distributed GPU test |
| Reference collector | Local Git export, license inclusion, file hashing and commit-lock reuse; no network used |

The maximum gradient difference in the bridge toy was approximately5.55e-17 in float64. This does not predict bf16 Qwen tolerances or prove the future implementation correct.

Results are in `validation/algebra_results.json`. The validator is a specification sanity check, not the test suite the coding agent must build for the real project.

Additional packaging checks validate relative Markdown links, source-snapshot hashes, Python syntax of included utilities, document presence, source-reference identifiers and ZIP integrity. See `validation/package_checks.json` for the final packaging result.

## Explicitly not verified here

- The local R2R JSON's actual Uniform8 sampling, instruction metadata, complete episodes or image files.
- Real Qwen tokenization, logits/gradient parity, generation-cache continuation, or checkpoint loading.
- The actual GPU/CUDA/DeepSpeed/kernel environment, FP32 reference-engine memory fit, or distributed training.
- Real VGGT feature grids, teacher/student alignment at runtime, or their navigation effect.
- Simulator action semantics, SR/SPL/NE, model latency, training throughput or scientific novelty/performance.
- Network fetching by the source collector. Web-visible reference files were inspected, but external repositories were not cloned in this environment. The local collector test does not establish network availability or permanent branch contents.

The actual thirteen-file input snapshot is preserved with hashes. Its original bugs and old environment reference remain visible as historical input; the version plans explain how to replace them. Do not launch that snapshot as though the planned fixes had already been applied.
