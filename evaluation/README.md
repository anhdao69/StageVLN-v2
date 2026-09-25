# R2R `val_unseen` evaluation artifacts

The complete runs below use the same Habitat R2R-VLNCE 0.2.4 configuration:
1,839 episodes, 500-step cap, seed 42, greedy single-action decoding, and
invalid-output fallback to `STOP`. The saved `summary.json` in each checkpoint
directory is the authoritative aggregate. `episodes.jsonl` contains per-episode
metrics. `trajectories.tar.gz` contains all per-episode JSON trajectories; extract
it in that checkpoint directory to recreate `trajectories/`.

| Checkpoint | History | Episodes | Success | SPL |
| --- | --- | ---: | ---: | ---: |
| `StageVLN-v0-r2r-uniform4-4gpu` | Uniform4 | 1,839 | 0.3899 | 0.3543 |
| `StageVLN-v1-r2r-sw4` | Sliding Window 4 | 1,839 | 0.3562 | 0.3018 |
| `StageVLN-v2_mem64_r0-r2r-epoch1` | Memory64, current | 1,839 | 0.1523 | 0.1395 |
| `StageVLN-v2_mem64_r0-r2r-epoch2` | Memory64, current | 1,839 | 0.1474 | 0.1296 |
| `StageVLN-v3_mem64_r4-r2r` | Memory64, Recent4 | 1,839 | 0.3235 | 0.2783 |

The Uniform8 v0 full run was still in progress when this branch was first
published. Its separate smoke artifact is included, but its partial benchmark
metrics are deliberately not listed as a completed result.

Downloaded model weights are not stored in Git; the checkpoint identity and
SHA-256 fingerprints are recorded in each result's `identity` block.
