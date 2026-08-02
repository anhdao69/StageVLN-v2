# SpatialForcing-VLN

Training-only spatial representation alignment for vision-language navigation with Qwen3.5, frozen VGGT, and JanusVLN trajectories.

SpatialForcing-VLN teaches a navigation model to internalize geometric information without adding a geometry encoder to its inference path. During training, the current observation is processed by a frozen VGGT teacher and aligned with the corresponding Qwen decoder features. During evaluation and deployment, the checkpoint runs as an ordinary Qwen3.5 model.

## Method

```text
JanusVLN history + current frame ──> Qwen3.5-4B ──> navigation action
                                          │
                              current-image layer-24 tokens
                                          │
                       LayerNorm -> Linear -> GELU -> Linear
                                          │
Current frame ──> frozen VGGT layer 23 ───┘
                         cosine alignment
```

The training objective is:

```text
total_loss = navigation_loss + 0.3 * spatial_forcing_loss
spatial_forcing_loss = mean(1 - cosine(projected_student, frozen_teacher))
```

The implementation deliberately does not use geometry fusion, a spatial cache, or historical frames in the teacher branch.

## Highlights

- Preserves JanusVLN prompts, ordered trajectory frames, and four navigation actions.
- Aligns only the final/current image tokens; history, prompt, action, and padding tokens are excluded.
- Extracts Qwen decoder layer 24 and VGGT aggregator layer 23.
- Resizes VGGT patch features as a two-dimensional grid before token alignment.
- Optionally adds the reference Spatial-Forcing aspect-ratio-aware UV sine/cosine encoding to VGGT features before resizing.
- Keeps VGGT frozen, in evaluation mode, and under `torch.no_grad()`.
- Logs navigation loss, alignment loss, total loss, cosine similarity, token counts, and gradient invariants.
- Saves no VGGT parameters in the Hugging Face checkpoint.
- Loads the trained navigation checkpoint through stock `Qwen3_5ForConditionalGeneration` for teacher-free inference.

## Verified configuration

The reference smoke test used Qwen3.5-4B, VGGT-1B, 4,096 R2R records, and 100 optimizer steps on four H100 GPUs. It predates the optional UV positional-embedding flag and therefore corresponds to `SF_USE_VGGT_PE=False`.

| Metric | Step 1 | Step 100 |
|---|---:|---:|
| Navigation loss | 2.1663 | 0.1861 |
| Spatial Forcing loss | 1.0211 | 0.5736 |
| Mean cosine similarity | -0.0211 | 0.4264 |
| Total loss | 2.4726 | 0.3582 |

The run completed in 563.1 seconds of trainer time (10 minutes 34 seconds end to end). Every logged step had nonzero Qwen/projector gradients, no VGGT gradients, and matching student/teacher token counts.

## Repository layout

```text
configs/spatial_forcing_vln_r2r.yaml       reference configuration
configs/spatial_forcing_vln_full.yaml      one-epoch R2R+RxR configuration
configs/datasets/                          editable annotation/media path pairs
scripts/train/train_spatial_forcing_vln.sh distributed training launcher
scripts/train/train_spatial_forcing_vln_full.sh one-epoch full-data recipe
scripts/validation/                        data, backward, and inference checks
src/qwen_vl/data/                          JanusVLN adapter and token masks
src/qwen_vl/model/spatial_forcing.py       projector and alignment objective
src/qwen_vl/train/                         model setup and distributed metrics
tests/                                     focused unit tests
```

The JanusVLN and Spatial-Forcing reference repositories are not vendored here. Model weights, datasets, checkpoints, papers, and local experiment reports are also excluded.

## Requirements

The tested environment uses:

```text
Python                    3.12.13
PyTorch                   2.10.0+cu129
torchvision               0.25.0+cu129
Transformers              5.3.0
DeepSpeed                 0.16.4
FlashAttention            2.8.3
Flash Linear Attention    0.5.2
Triton                    3.7.1
```

CUDA 12.9 and an Ampere-or-newer NVIDIA GPU are recommended. Multi-GPU training was verified on H100.

## Installation with uv

```bash
git clone https://github.com/anhdao69/SpatialForcing-VLN.git
cd SpatialForcing-VLN

uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu129

uv pip install psutil ninja wheel setuptools packaging
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install causal-conv1d==1.6.2.post1 --no-build-isolation
uv pip install flash-linear-attention==0.5.2 --no-build-isolation
uv pip install -e ".[dev]"
```

PyTorch 2.10 installs Triton 3.6 by default. That version is affected by the Qwen3.5/FLA gated-delta backward restriction on Hopper. For H100, apply the verified override after installing PyTorch:

```bash
uv pip install --no-deps triton==3.7.1
```

The training launcher checks this condition before allocating model replicas and prints the remediation command when necessary.

## Data preparation

Download or construct the JanusVLN trajectory data using the upstream [JanusVLN repository](https://github.com/MIV-XJTU/JanusVLN). Datasets use SpatialStack's annotation/media path-pair convention:

```json
{
  "dataset_name": "janusvln_r2r_rxr",
  "annotation_path": "/mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json",
  "data_path": "/mnt/data/vmo-ai-task/anhdh35/JanusVLN",
  "tag": "3d",
  "dataset_format": "janusvln"
}
```

To switch datasets, edit the path pair or set `DATASET_CONFIG` to another JSON file. A config may also contain a list of named entries; set `DATASET_USE` to select a comma-separated subset. The included R2R test configuration has this layout:

```text
JanusVLN_data/
├── train_r2r.json
└── R2R-CE-640x480/
    └── train/
        └── <episode-id>/
            └── <trajectory-frame>.png
```

Each JSON record must contain 1-9 ordered image paths and a final action in:

```text
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
STOP
```

The final image is treated as the current observation; preceding images are history. Image paths may be relative to the data root. Legacy absolute paths are normalized from their `R2R-CE-640x480` component.

## Validate the data adapter

This command checks real records with 1, 2, 4, and 9 frames and verifies that only the final image span is selected:

```bash
python scripts/validation/validate_spatial_forcing_data.py \
  --model-path Qwen/Qwen3.5-4B \
  --dataset-config configs/datasets/janusvln_r2r.json
```

For the tested 640×480 R2R observations, Qwen produces grid `[1, 24, 32]`, corresponding to 192 tokens after its 2×2 spatial merge.

## Validate one complete backward pass

```bash
python scripts/validation/validate_spatial_forcing_forward.py \
  --model-path Qwen/Qwen3.5-4B \
  --teacher-path facebook/VGGT-1B \
  --data-root /path/to/JanusVLN_data \
  --cache-dir /path/to/model-cache \
  --use-vggt-pe
```

The validator requires:

- finite navigation, Spatial Forcing, and total losses;
- equal student and resized-teacher position counts;
- a nonzero projector gradient;
- a nonzero Qwen student gradient;
- no VGGT gradient.

## Training

The launcher defaults to four important settings from the first-stage recipe: batch size 1 per GPU, gradient accumulation 8, bf16, and DeepSpeed ZeRO-2.

```bash
MODEL_PATH=Qwen/Qwen3.5-4B \
TEACHER_MODEL_PATH=facebook/VGGT-1B \
JANUSVLN_DATA_ROOT=/path/to/JanusVLN_data \
CACHE_DIR=/path/to/model-cache \
OUTPUT_DIR=/path/to/output \
NPROC_PER_NODE=4 \
bash scripts/train/train_spatial_forcing_vln.sh
```

Useful overrides:

| Variable | Default | Purpose |
|---|---:|---|
| `NPROC_PER_NODE` | visible GPU count | Distributed world size |
| `DATASET_CONFIG` | unset | JSON object/list with `annotation_path` and `data_path` |
| `DATASET_USE` | config entries or `janusvln_r2r` | Optional named config selection |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Steps accumulated per GPU |
| `NUM_TRAIN_EPOCHS` | `1` | Epoch count when `MAX_STEPS` is unset |
| `LEARNING_RATE` | `1e-6` | Qwen and multimodal merger learning rate |
| `SF_PROJECTOR_LR` | `1e-5` | Alignment projector learning rate |
| `SF_LOSS_WEIGHT` | `0.3` | Auxiliary-loss weight |
| `SF_USE_VGGT_PE` | `False` | Add reference UV positional encoding to VGGT features before pooling |
| `MAX_STEPS` | unset | Optional positive bound for smoke tests only |
| `MAX_SAMPLES` | `-1` | Optional JSON prefix size |
| `DATALOADER_NUM_WORKERS` | `4` | Workers per distributed process |
| `SAVE_STRATEGY` | `steps` | Hugging Face checkpoint strategy |
| `SAVE_STEPS` | `1000` | Optimizer-step interval between checkpoints |
| `SAVE_TOTAL_LIMIT` | `10` | Number of periodic checkpoints retained; `0` keeps all |
| `WARMUP_STEPS` | `1` | Optimizer warmup |
| `WARMUP_RATIO` | unset | Ratio-based warmup; takes precedence when set |

For a 100-step smoke test without large ZeRO optimizer checkpoints:

```bash
MAX_STEPS=100 \
MAX_SAMPLES=4096 \
SAVE_STRATEGY=no \
SF_USE_VGGT_PE=True \
MODEL_PATH=Qwen/Qwen3.5-4B \
TEACHER_MODEL_PATH=facebook/VGGT-1B \
JANUSVLN_DATA_ROOT=/path/to/JanusVLN_data \
OUTPUT_DIR=./output/r2r-smoke \
NPROC_PER_NODE=4 \
bash scripts/train/train_spatial_forcing_vln.sh
```

The launcher writes periodic `checkpoint-<step>` directories according to the save controls. It also writes the final student model, processor/tokenizer, `trainer_state.json`, and `train.log` directly to `OUTPUT_DIR`. For `T` optimizer steps, step-based saving triggers `floor(T / SAVE_STEPS)` periodic saves; at most `SAVE_TOTAL_LIMIT` remain when the limit is positive.

### Full R2R + RxR training

The full launcher consumes every record for exactly one epoch, enables the VGGT UV positional encoding, uses a 3% warmup ratio, saves every 1,000 optimizer steps by default, retains the latest 10 periodic checkpoints, and deliberately does not pass `max_steps`:

```bash
NPROC_PER_NODE=4 \
CACHE_DIR=/path/to/model-cache \
OUTPUT_DIR=/path/to/output \
bash scripts/train/train_spatial_forcing_vln_full.sh
```

For example, save every 500 optimizer steps and retain the latest 20 checkpoints:

```bash
SAVE_STEPS=500 \
SAVE_TOTAL_LIMIT=20 \
NPROC_PER_NODE=4 \
bash scripts/train/train_spatial_forcing_vln_full.sh
```

The validated R2R dataset has 19,727 optimizer steps per epoch on four GPUs with gradient accumulation 8. At the defaults, 19 periodic saves are triggered, the latest 10 numbered checkpoints remain, and the final step-19,727 model is written directly to the output root.

It defaults to `configs/datasets/janusvln_r2r_rxr.json`. To use another full annotation without changing code:

```bash
DATASET_CONFIG=/path/to/my_dataset.json \
NPROC_PER_NODE=4 \
bash scripts/train/train_spatial_forcing_vln_full.sh
```

## Teacher-free checkpoint validation

The training checkpoint intentionally omits all frozen VGGT tensors. Verify a saved directory using only the stock Qwen class:

```bash
python scripts/validation/validate_spatial_forcing_inference.py \
  --checkpoint /path/to/output \
  --data-root /path/to/JanusVLN_data
```

The check asserts that VGGT, the projector, and geometry-fusion modules are not initialized and that a real R2R navigation loss is finite. Stock Transformers may report the six saved projector tensors as ignored keys; storing them permits later Spatial Forcing training to retain the learned projector, but they are not instantiated or executed by stock Qwen inference.

## Tests

```bash
pytest -q tests
```

The focused suite covers JSON streaming, path normalization, current-frame masks, incorrect token counts, two-dimensional grid resizing, cosine loss, and gradient isolation.

## Implementation notes

- Hugging Face hidden-state index 24 corresponds to the output of zero-based Qwen decoder block 23. Transformers 5.3 does not populate the intermediate hidden-state tuple in this Qwen3.5 training path, so the implementation captures that exact block output with a temporary forward hook.
- The verified current frame gives 768 raw VGGT patches (`24×32`) and 192 merged Qwen positions (`12×16`). VGGT is reshaped spatially and bilinearly resized to `12×16`; the flattened sequence is never interpolated directly.
- `SF_USE_VGGT_PE=True` adds the reference implementation's normalized UV sine/cosine grid at scale `0.1` to the frozen VGGT grid before bilinear resizing. `False` preserves previously verified checkpoints and training behavior. VGGT's own learned positional embedding and 2D RoPE remain active in both modes.
- `use_geometry_encoder` and `use_geometry_fusion` must both be false. The VGGT instance owned by Spatial Forcing is loss-only.
- The full Qwen language model and multimodal merger are trainable; the Qwen vision tower and VGGT are frozen.

## Acknowledgements

This project builds on:

- [SpatialStack](https://github.com/jzh15/SpatialStack), the Qwen3.5/VGGT training foundation;
- [Spatial Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing), the representation-alignment idea;
- [JanusVLN](https://github.com/MIV-XJTU/JanusVLN), the VLN trajectory and prompt setup;
- [VGGT](https://github.com/facebookresearch/vggt) and [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B).

Please cite the corresponding upstream works when using this code in research.

## License

Released under the [Apache License 2.0](LICENSE). Upstream components retain their original notices and licenses.
