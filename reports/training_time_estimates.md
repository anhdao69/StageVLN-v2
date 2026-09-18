# R2R training time on four H100s

Measured 2026-09-18 in the existing interactive allocation826441. These are
one-epoch estimates for global batch 64, based on successful bounded runs;
no full epoch or new Slurm submission was performed.

| Version | Selected per-GPU reader batch | Accumulation / temporal policy | Measured epoch estimate | Planning allowance | Prepared Slurm limit |
|---|---:|---|---:|---:|---:|
| v0 Uniform8 + current | 2 | 8 SFT microbatches/update | 21.2–21.9 h | About22–25 h | 28 h |
| v1 Recent4 + current | 4 | 4 SFT microbatches/update | 8.4–9.0 h | About9–11 h | 16 h |
| v2 Memory64 + current | 4 | K8; actual64-label global update | 5.92 h | About6–8 h | 12 h |
| v3 Memory64 + Recent4 + current | 8 | K8; actual64-label global update | 9.43 h | About10–13 h | 16 h |

All wrappers request eight CPUs, one image-preprocessing worker per GPU, and
one math thread per process. v0/v1 use DeepSpeed ZeRO-1 with the original BF16
SFT recipe and no activation checkpointing. v2/v3 use native PyTorch optimizer
state sharding, FP32 parameters/gradients/Adam/state, BF16 compute, and the exact
reader-gradient bridge. v2 disables checkpointing; v3 enables it. The different
engines, numerical recipes, image counts, and data order mean this table is a
runtime comparison, not evidence of a memory-quality improvement.

The planning allowances provide room for checkpoint/export I/O and variable
throughput. Queueing and Hugging Face upload time are additional and unmeasured.
There is no guaranteed walltime under arbitrary cluster/filesystem contention.
The prepared v0 limit was increased from 24 to28 hours after measuring it; the
normal QoS and highgpu partition reported no maximum walltime restriction.

## Original v0/v1 measurements

Both ran32 updates over the same2,048 unique states sampled uniformly without
replacement from the entire631,244-state R2R corpus, seed 42. The selected row
indices, annotation hashes, and image counts are in
[the subset record](../implementations/baseline_timing_subset.json).
Average images/state were8.4009 for v0 and 4.8481 for v1. The model/trainer source
was main `7a76695`; the later main commit `c340227` only increases v0 walltime.

| Measurement | v0 | v1 |
|---|---:|---:|
| Trainer runtime,32 updates | 255.2227 s | 105.2712 s |
| Average seconds/update including warmup | 7.9757 | 3.2897 |
| Approximate steady seconds/update, excluding first2 | 7.7333 | 3.0667 |
| First / last logged loss | 1.764 /0.1829 | 1.536 /0.1722 |
| Rank0 peak allocated memory | 65.544 GiB | 70.944 GiB |

All32 logged loss/gradient norms are finite, and the mean loss over the final
four updates is lower than the first four. These are different training states
across updates, not an overfit or navigation-quality test.

The steady estimates use integer-second progress timestamps at updates 2 and 32,
so they have one-second timestamp resolution. The upper estimates use the
unadjusted Trainer runtime/32 and retain the short run's warmup overhead.
The full SFT epoch has 9,864 updates and 631,248 exposures: Accelerate repeats4
states to balance the final batches, giving a final16-state update. Formula:
`seconds_per_update * 9864 / 3600`.

[Exact metrics](../implementations/baseline_timing_results.json),
[v0 log](../implementations/benchmark_v0_timing32.log), and
[v1 log](../implementations/benchmark_v1_timing32.log) supersede estimates from
the earlier very short stress runs. These timing-only runs disable checkpoints
and final model saving. Their shortened cosine schedules do not reproduce the
full-epoch learning-rate prefix.

## Recurrent measurements and matched controls

Each primary recurrent timing uses32 updates/2,048 states from the first64
complete episodes. The recorded mean excludes the first two full updates;
timings take the maximum across ranks. The complete corpus has 631,244 real
states,9,864 updates, and a final12-label update without padding.

The selected v2/v3 long-instruction stress peaks are67.983/53.661 GiB allocated
and 69.455/58.988 GiB reserved. Both finish all 1,165 stress states, preserving
K8 and native production image bounds. This is measured fit for the audited
workload, not a universal memory bound.

The matched chronological no-memory controls also ran32 updates each:
current-only_ep estimates4.00 h and Recent4_ep at batch 8 estimates9.01 h.
These controls match the recurrent engine/precision/data order; they are
separate from the original IID v0/v1 rows above.

See [the detailed report](v2_v3_implementation_report.md),
[machine-readable recurrent timings](../implementations/recurrent_performance.json),
and [validation commands](validation_commands.md) for numerical checks,
source snapshots, checkpoint locations, and remaining simulator/full-training work.
