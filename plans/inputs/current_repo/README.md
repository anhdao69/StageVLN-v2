# Qwen3.5 R2R SFT

This repository contains one training path: plain supervised fine-tuning of
`Qwen/Qwen3.5-4B` on JanusVLN R2R. The vision encoder is frozen; the language
model and Qwen vision merger are trained. There is no geometry model, auxiliary
loss, distillation, depth supervision, or inference-time modification.

Each record may contain the current observation and up to eight ordered history
frames. The final image is the current observation. The expected actions are
`MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, and `STOP`.

## Train

The Newton paths are in
[`configs/datasets/newton_r2r_uniform8.json`](configs/datasets/newton_r2r_uniform8.json).
The launcher activates the uv environment at `../SpatialForcing-VLN/.venv`.

```bash
bash train/v0_uniform8.sh
```

The defaults retained from the previous SFT recipe are:

- Qwen3.5-4B, bf16, TF32, and PyTorch SDPA
- language model and multimodal merger trainable; vision encoder frozen
- batch size 1 per GPU and gradient accumulation 8
- language-model learning rate `1e-6` and merger learning rate `1e-5`
- cosine schedule, one warmup step, weight decay `0.01`, and ZeRO-2
- one epoch, 12,800-token limit, and eight history frames

Gradient checkpointing defaults to off on H100 for higher throughput. Enable it
only if a longer example exceeds memory:

```bash
GRADIENT_CHECKPOINTING=True bash train/v0_uniform8.sh
```

For a bounded smoke run without checkpoint writes:

```bash
MAX_STEPS=1 MAX_SAMPLES=32 SAVE_STRATEGY=no SAVE_FINAL_MODEL=False \
  bash train/v0_uniform8.sh
```

The launcher writes Hugging Face training metrics to `train_results.json` and
prints end-to-end wall time to `train.log`. Paths and run controls can be
overridden with `DATASET_CONFIG`, `CACHE_DIR`, `OUTPUT_DIR`, `NPROC_PER_NODE`,
`MAX_STEPS`, and the other environment variables defined near the top of the
launcher.

On four H100 80GB GPUs, a two-step benchmark at the default global batch size
of 32 took 12.11 seconds of trainer time (6.06 seconds/step). With 631,264 R2R
examples (19,727 steps), that projects to about 33.2 hours for one epoch. This
is a throughput estimate; filesystem load and trajectory-length variation can
change the full-run wall time.
