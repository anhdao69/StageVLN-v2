# SpatialForcing v4: Depth Pretraining Validation

Date: 2026-08-19
Branch: `v4`
Validation dataset: `janusvln_r2r` from
`configs/datasets/janusvln_r2r.json`

## Outcome

The real-data validation passed on 100 R2R examples (annotation indices
100–199) using the actual cached Qwen3.5-4B and VGGT-1B checkpoints on one H100.
Every example completed the combined navigation + Spatial Forcing + depth
forward path with:

- one Qwen forward;
- exactly one VGGT aggregator execution;
- the same current RGB field of view for Qwen and the dense teacher target;
- exactly 192 current-image Qwen tokens (`12 x 16` merged grid);
- exactly 1,036 raw VGGT tokens (`28 x 37` padded teacher grid);
- a 384 x 512 student/teacher depth comparison;
- no NaN, Inf, or non-positive VGGT depth values;
- finite positive student predictions; and
- layer-24 reuse between the v0 SF branch and the depth branch.

The machine-local combined R2R+RxR annotation was also inspected. It contains
`parquet:` media references, but the referenced parquet files are not installed
under the configured data root. The checked-in combined config instead points
to an unavailable `/mnt/data/...` tree from another machine. Consequently, RxR
validation was not claimed or fabricated; a valid RxR media/config installation
is still required before the full combined experiment.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/scratch/11528/anhdao69/model_cache \
.venv/bin/python scripts/validation/validate_depth_supervision.py \
  --model-path /scratch/11528/anhdao69/model_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --teacher-path /scratch/11528/anhdao69/model_cache/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9 \
  --dataset-config configs/datasets/janusvln_r2r.json \
  --dataset-use janusvln_r2r \
  --output-dir reports/v4_validation_r2r_100 \
  --cache-dir /scratch/11528/anhdao69/model_cache \
  --max-samples 100 \
  --start-index 100 \
  --visualizations 4
```

## Aggregate statistics

| Metric | Mean | Min across examples | Max across examples |
|---|---:|---:|---:|
| VGGT aggregator calls | 1.0000 | 1.0000 | 1.0000 |
| current Qwen tokens | 192.0000 | 192.0000 | 192.0000 |
| teacher depth valid fraction | 1.0000 | 1.0000 | 1.0000 |
| teacher depth minimum | 0.298406 | 0.129138 | 0.592480 |
| teacher depth maximum | 2.169197 | 1.203877 | 4.210157 |
| teacher depth mean | 0.858483 | 0.812952 | 0.986618 |
| untrained student depth mean | 0.999507 | — | — |
| raw GeoVR-style depth loss | 0.450335 | 0.264769 | 0.729057 |
| depth L1 regression | 0.337888 | — | — |
| depth gradient loss | 0.112448 | — | — |
| depth MAE | 0.352630 | 0.172170 | 0.630209 |
| weighted depth (`0.05 * L_depth`) | 0.022517 | — | — |
| SF loss | 0.976759 | — | — |
| weighted SF (`0.3 * L_SF`) | 0.293028 | — | — |

Across all 100 examples:

```text
teacher NaN fraction         = 0
teacher Inf fraction         = 0
teacher non-positive fraction= 0
all recorded scalar values finite
```

The VGGT compatibility padding was constant across this R2R subset:

```text
Qwen/current field of view: 384 x 512
VGGT internal padded canvas: 392 x 518
bottom padding:                 8 pixels
right padding:                  6 pixels
comparison target:           384 x 512 (cropped back to Qwen FOV)
```

## Visualizations

The generated panels are:

- `reports/v4_validation_r2r_100/depth_sample_100.png`
- `reports/v4_validation_r2r_100/depth_sample_101.png`
- `reports/v4_validation_r2r_100/depth_sample_102.png`
- `reports/v4_validation_r2r_100/depth_sample_103.png`

Each panel contains current RGB, VGGT pseudo-depth, student depth, and absolute
error using the same teacher color range. The VGGT maps visibly follow walls,
doorways, floor, ceiling, furniture, and distant openings without a field-of-
view shift. The randomly initialized student is close to depth 1, as expected
from GeoVR's final exponential output and default PyTorch initialization.

After the 100-step smoke run, the same four examples were reloaded from the
saved checkpoint and rendered under:

- `reports/v4_validation_r2r_trained_step100/`

The trained student remained positive and finite and began learning broad depth
structure. On these four examples its mean SF cosine was 0.410217, mean student
depth was 0.854739, mean depth loss was 0.499116, and the one-aggregator
invariant remained exact.

## Interpretation

The startup depth scale supports the planned `depth_loss_weight=0.05`:

```text
mean raw depth loss       = 0.450335
mean weighted depth loss  = 0.022517
mean weighted SF loss     = 0.293028
```

Depth is therefore meaningful but not dominant at initialization. No initial
lambda sweep is warranted. The full 100-step smoke results and final lambda
decision are recorded in `reports/v4_implementation_report.md`.
