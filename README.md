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

The SFT defaults are:

- Qwen3.5-4B, bf16, TF32, and FlashAttention 2
- language model and multimodal merger trainable; vision encoder frozen
- batch size 2 per GPU and gradient accumulation 4 (global batch 32 on 4 GPUs)
- language-model learning rate `1e-6` and merger learning rate `1e-5`
- cosine schedule, one warmup step, weight decay `0.01`, fused AdamW, and ZeRO-1
- one epoch, 12,800-token limit, and eight history frames

Qwen3.5 applies multimodal RoPE before attention. The training entry point works
around a Transformers 5.3 bug that incorrectly treats Qwen's three-axis
position IDs as FlashAttention packed-sequence metadata. The collator also asks
the stock Qwen loss for only the assistant-action suffix logits; ignored prompt
and image-token logits are skipped without changing the SFT loss.

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

On four H100 80GB GPUs, the final 20-step smoke benchmark trained 640 samples
in 57.71 seconds: 11.09 samples/s and 2.89 seconds per global-batch-32 step,
including warm-up. At that compute rate, 631,264 examples (19,727 steps) take
about 15.8 hours, versus the original SDPA/ZeRO-2 estimate of 33.2 hours. This
is a compute-throughput estimate on repeated nine-frame samples; filesystem
load, image diversity, and trajectory-length variation can increase full-run
wall time.
