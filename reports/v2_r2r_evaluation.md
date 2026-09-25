# v2 recurrent-memory Habitat R2R evaluation

## Scope and status

This evaluation loads the immutable epoch-1 and epoch-2 policy exports under
`checkpoints/`, runs closed-loop R2R-VLNCE `val_unseen`, and writes resumable
results under `evaluation/r2r_v2/<checkpoint>/`. The checkpoints are evaluated
sequentially on one GPU. Epoch 1 is evaluated first; epoch 2 starts after epoch
1 completes. A partial summary is not a final benchmark result.

The full split contains 1,839 episodes. The first observed policies can reach
the 500-action cap, so the complete two-checkpoint run can take tens of hours.
The evaluator appends one episode at a time and can be relaunched safely with
`scripts/eval/run_r2r_epochs.sh`.

## Train/eval alignment audit

| Contract | Training | Habitat evaluation | Status |
|---|---|---|---|
| Observation/action order | Frame `o_t` is paired with expert action `a_t` | Render RGB, write it once, predict, then execute action | Aligned |
| Memory timing | `M_t = writer(M_(t-1), o_t, instruction)` before reader loss | `PolicySession.observe` writes current RGB before generation | Aligned |
| Reader images | v2 uses current image only (`recent=0`) | No RGB history is retained by Qwen; only recurrent memory persists | Aligned |
| Episode state | Fresh memory at episode start; carry across chronological steps | `reset` per episode; contiguous step IDs; no cross-episode cache | Aligned |
| Prompt | Shared non-thinking Qwen template and exact v0 Janus wording | Uses exported tokenizer plus the same `build_state(..., include_target=False)` | Aligned |
| Image preprocessing | Qwen processor, min 12,544 and max 451,584 pixels | Loads exported processor and checks effective bounds and all recorded fields | Aligned |
| Decoding | Greedy, fresh Qwen cache per action, exact primitive label | Same actor generation; 16-token cap; no substring matching | Aligned |
| Invalid output | Strict decoder records failure; master plan declares STOP fallback | Records raw invalid output, commits one memory write, executes STOP | Aligned and explicit |
| Action IDs | `STOP=0`, `MOVE_FORWARD=1`, `TURN_LEFT=2`, `TURN_RIGHT=3` in source R2R GT | Same Habitat primitive mapping | Aligned |
| Camera/control | Training audit reports 640×480 first frames from R2R-VLNCE | 640×480 RGB, 79° HFOV, 0.25 m forward, 15° turns | Consistent with supplied evaluator; full training pixels are unavailable locally for byte-level comparison |
| Metrics | Not used for SFT | Habitat SR/SPL/NE plus trajectory-minimum OS at 3 m | Benchmark-only, no train leakage |

The evaluator rejects non-v2 exports, smoke-only exports, architecture changes,
prompt action-order changes, and processor mismatches before model rollout. The
epoch exports have different memory hashes and different backbone weight bytes,
so epoch 1 and epoch 2 are not aliases. Their exported run manifests are
otherwise identical, as expected for successive epochs of one five-epoch run.

## Benchmark protocol

- Dataset: R2R-VLNCE v1-3 preprocessed `val_unseen`, 1,839 episodes.
- Simulator: Habitat-Lab/Habitat-Sim 0.2.4.
- Sensor: 640×480 RGB, 79° horizontal field of view.
- Controller: 0.25 m forward and 15° turns.
- Success radius: 3 m.
- Episode cap: 500 executed actions; the final action is forced to STOP so
  Habitat computes STOP-dependent success consistently.
- Policy: deterministic BF16 inference with deterministic FlashAttention 2,
  greedy exact-label decoding, and no oracle stop.
- Invalid generations: reported and executed as STOP; no free-text substring
  recovery is allowed.

`oracle_success` is one if any recorded distance-to-goal is strictly below 3 m.
`navigation_error` is the terminal distance. SPL and success are read directly
from Habitat after the executed STOP/terminal transition.

## Result and trajectory layout

Each checkpoint directory contains:

- `episodes.jsonl`: one metric record per completed episode;
- `summary.json`: aggregate over all currently completed episode records;
- `trajectories/<scene>__<episode>.json`: one atomic trajectory per metric row.

Each trajectory stores the instruction, initial agent position/quaternion and
distance, then every predicted action, executed action, raw model text,
invalid-output flag, post-action position/quaternion, and distance-to-goal. Its
outcome block repeats SR/SPL/OS/NE and step count. `episodes.jsonl` contains the
relative `trajectory_file` path. On resume, completed metrics without the
corresponding trajectory are rejected rather than silently skipped.

## Training evidence and limits

The exports record v2 Memory64/R0, K=8 chronological training, global labeled
state batch 64, four ranks, BF16 autocast, FP32 trainable/Adam state, bridge
execution, and exact processor/prompt fingerprints. Existing tests establish
adapter parity, direct/bridge temporal gradient parity, chronological reset and
resume behavior, and actor prefix parity. The five-epoch runner tests also
establish continuous optimizer/scheduler state across epoch boundaries.

The portable exports do not include `metrics.jsonl`, optimizer shards, epoch
completion markers, or the original training images on this host. Therefore,
training-loss curves and exact completed-update counts cannot be independently
reconstructed from these export directories alone. Closed-loop evaluation must
be used to select between epoch 1 and epoch 2; lower teacher-forced CE should
not be assumed to imply higher SR/SPL.
