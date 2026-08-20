# Spatial Forcing for SpatialStack VLN: implementation and verification report

Date: 2026-08-01<br>Plan implemented: `plans/v0.md`<br>Status: implemented and verified with a clean 100-step, 4-H100 R2R run

## 1. Result

SpatialStack now has a loss-only Spatial Forcing training path for Qwen3.5-4B. It uses the JanusVLN R2R history/current-frame prompt setup, supervises Qwen decoder layer 24 with frozen VGGT layer 23 features from only the current frame, and optimizes

```texttotal_loss = navigation_loss + 0.3 * spatial_forcing_loss```

VGGT features are never fused into Qwen. The trained checkpoint was loaded and evaluated with the stock `Qwen3_5ForConditionalGeneration` class. That inference process initialized no VGGT, Spatial Forcing projector, geometry encoder, geometry merger, or spatial cache.

The final 100-step smoke run completed with exit status 0. Its Spatial Forcing loss decreased from 1.0211 to 0.5736, while mean cosine similarity increased from -0.0211 to 0.4264. Navigation loss remained finite and decreased from 2.1663 to 0.1861.

## 2. Design implemented

### JanusVLN R2R data path

The adapter reads `/scratch/11528/anhdao69/data/JanusVLN_data/train_r2r.json` and resolves image names relative to the configured JanusVLN root. Legacy absolute image names are normalized from their `R2R-CE-640x480` component, so the annotations are relocatable.

The adapter preserves:

- the source JanusVLN conversation text;- action labels `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, and `STOP`;- source frame order;- one to eight history frames followed by the current frame;- the final frame as the current observation.

It emits the normal Qwen tensors plus ordered frame paths, frame count, current-frame index, current-image token mask/grid, and exactly one current-frame VGGT tensor. Historical frames are processed by Qwen but are not sent to VGGT.

For smoke tests, the JSON-array reader can stop after `max_samples` records instead of materializing the full approximately 1.1-GB annotation file on every distributed rank.

### Current-image masking

The mask is built from the final explicit `<|vision_start|> ... <|vision_end|>` span and selects only `<|image_pad|>` tokens inside that span. It does not assume a fixed suffix length. The dataset and collator assert that:

- the current frame index is `frame_count - 1`;- the mask is not truncated by `model_max_length`;- its token count equals the final Qwen image grid after spatial merging.

Real R2R records containing 1, 2, 4, and 9 frames were validated. Every current image had Qwen grid `[1, 24, 32]` and 192 post-merge tokens; no historical image tokens entered the mask.

### Student representation

The requested student representation is Hugging Face `hidden_states[24]`, meaning the output of zero-based decoder block 23. Transformers 5.3.0's Qwen3.5 implementation accepts hidden-state arguments but does not populate the intermediate tuple in this training path. I therefore capture the output of `model.language_model.layers[23]` with a temporary forward hook. This is the exact tensor corresponding to `hidden_states[24]` under Hugging Face's embedding-at-index-zero convention. The hook exists only during a labeled Spatial Forcing training forward and is removed in `finally`.

Only positions selected by the explicit current-image mask are aligned. Prompt, instruction, history-image, action, and padding positions are excluded.

### Teacher representation and grid matching

VGGT-1B is loaded as a separate registered training-only teacher with camera, point, depth, and tracking heads disabled. It is forced to evaluation mode, all parameters have `requires_grad=False`, and feature extraction runs under `torch.no_grad()`.

The selected VGGT layer is SpatialStack aggregator index 23. Only patch tokens are returned; camera tokens are excluded. For the verified R2R image shape:

```textVGGT current-frame input: 336 x 448VGGT patch size:          14raw VGGT grid:            24 x 32 = 768 tokensQwen merged grid:         12 x 16 = 192 tokens```

VGGT tokens are reshaped to a two-dimensional grid, bilinearly resized to the Qwen grid in float32, restored to the teacher dtype, and flattened. A hard assertion requires the final teacher and student position counts to match.

### Projector and loss

The trainable projector is:

```textLayerNorm(2560)Linear(2560 -> 4096)GELULinear(4096 -> 2048)```

Projected student and detached teacher features are normalized in float32. The auxiliary objective is the mean position-wise `1 - cosine_similarity`. The projector uses the existing `mm_projector_lr` optimizer group; the verified run used `1e-5`, matching the existing SpatialStack projector learning-rate convention. Qwen and its multimodal merger used the plan's base `1e-6` learning rate.

### No fusion and teacher-free inference

The training wrapper subclasses the stock Qwen3.5 conditional-generation model and contains no SpatialStack geometry-fusion module. Spatial Forcing startup and forward both reject `use_geometry_encoder=True` or `use_geometry_fusion=True`.

Frozen `spatial_teacher.*` weights are removed from the Hugging Face state dict. The saved model contains six projector tensors so Spatial Forcing training state is retained, but the stock Qwen class ignores those keys and does not create a projector. The verified final checkpoint contains:

```texttotal tensor keys:        730spatial_teacher keys:       0geometry keys:              0spatial_projector keys:      6```

## 3. Files changed

- `src/qwen_vl/data/__init__.py`: registered `janusvln_r2r` and root-relative configuration.- `src/qwen_vl/data/data_qwen.py`: bounded JSON-array reader, path normalization, JanusVLN validation, current-only teacher preparation, explicit mask metadata, collation, and invariants.- `src/qwen_vl/data/utils.py`: made geometry preprocessing optional so history frames do not create VGGT tensors.- `src/qwen_vl/model/spatial_forcing.py`: Qwen training wrapper, frozen teacher, layer capture, 2-D grid resize, projector, cosine loss, metrics, gradient diagnostics, and teacher-free state dict.- `src/qwen_vl/model/geometry_encoders/vggt_encoder.py`: cache-directory support and current PyTorch autocast API.- `src/qwen_vl/train/argument.py`: Spatial Forcing, fusion, layer, weight, projector, and JanusVLN-root arguments.- `src/qwen_vl/train/train_qwen.py`: validated loss-only model selection, teacher/projector trainability, Spatial Forcing trainer selection, rank-zero output, cache handling, and distributed teardown.- `src/qwen_vl/train/trainer.py`: distributed averages for separate objectives/invariants and projector optimizer grouping.- `scripts/train/train_spatial_forcing_vln.sh`: reproducible Qwen3.5-4B + frozen VGGT + ZeRO-2 launcher.- `configs/spatial_forcing_vln_r2r.yaml`: human-readable first-stage configuration.- `scripts/validation/validate_spatial_forcing_data.py`: real 1/2/4/9-frame mask validation.- `scripts/validation/validate_spatial_forcing_forward.py`: full Qwen+VGGT forward/backward validation.- `scripts/validation/validate_spatial_forcing_inference.py`: stock-Qwen-only checkpoint validation.- `tests/test_janusvln_spatial_forcing_data.py`: JSON, path, and final-span mask unit tests.- `tests/test_spatial_forcing_loss.py`: spatial resize, cosine objective, and gradient-isolation unit tests.

Unrelated pre-existing deletions under `src/lmms_eval` and untracked reference-code/paper folders were deliberately not changed.

## 4. Runtime and training configuration

### Software

```textPython                    3.12.13PyTorch                   2.10.0+cu129Transformers              5.3.0DeepSpeed                 0.16.4Triton                    3.7.1FlashAttention            2.8.3Flash Linear Attention    0.5.2```

PyTorch 2.10 normally installs Triton 3.6.0. FLA rejects that build for Qwen3.5 gated-delta backward on Hopper because of the affected Triton range. The verified uv environment uses:

```bashuv pip install --python .venv/bin/python --no-deps triton==3.7.1```

The training launcher now performs a Hopper/Triton preflight and prints this exact remediation before allocating model replicas. It also puts multiprocessing sockets and Triton caches on node-local temporary storage; this removed the shared-Lustre Python worker cleanup tracebacks seen during the preliminary 8-step run.

### Hardware and Slurm

```textSlurm job:        3364435partition:        h100node:             c561-009GPU count/type:   4 x NVIDIA H100, 95,830 MiB eachCPU allocation:   96 cores (48 assigned to the training step)precision:        bf16, TF32 enabledDeepSpeed:        ZeRO stage 2```

Observed GPU memory during steady-state training was 44,731-45,567 MiB per H100. Utilization at the sampled instant was 58-62%; it varies because image loading, Qwen, and VGGT phases alternate.

### Final smoke configuration

```textdataset:                       R2R prefix, 4,096 distinct recordsmax steps:                     100per-device batch:              1gradient accumulation:         8world size:                    4effective global batch:        32sample presentations:          3,200completed epoch fraction:      0.78125Qwen/merger learning rate:     1e-6projector learning rate:       1e-5schedule:                      cosine, 1 warmup stepSpatial Forcing weight:        0.3student/teacher layers:        24 / 23Qwen vision tower:             frozenQwen language + merger:        trainableVGGT:                          frozen, current frame onlygradient checkpointing:        enableddata flattening:               disabled```

`SAVE_STRATEGY=no` was used only for this smoke run to avoid writing approximately 59 GiB of ZeRO optimizer partitions after a 100-step check. The launcher default remains checkpointed step-based saving for full training.

### Timing

```textTrainer-reported training loop:     563.1 s (9 min 23.1 s)End-to-end launcher wall time:      10 min 34.35 sOptimizer-step throughput:          0.178 steps/sTraining-sample throughput:         5.683 samples/sFinal HF checkpoint write:          approximately 13.1 sFinal checkpoint size:              9.7 GiB```

End-to-end time includes four Qwen and four VGGT loads, four independent 4,096-record JSON-prefix parses, processor/tokenizer setup, distributed initialization, training, final state collection, and model serialization.

## 5. Training behavior

### Endpoint and windowed metrics

| Metric | Step 1 | Step 100 | First 10 mean | Last 10 mean ||---|---:|---:|---:|---:|| navigation loss | 2.1663 | 0.1861 | 0.6932 | 0.1880 || Spatial Forcing loss | 1.0211 | 0.5736 | 0.9953 | 0.5777 || mean cosine similarity | -0.0211 | 0.4264 | 0.0047 | 0.4223 || total loss | 2.4726 | 0.3582 | 0.9918 | 0.3613 |

The first-to-last-10-window Spatial Forcing decrease was 0.4176 (42.0%). The corresponding cosine increase was 0.4176. A least-squares fit across all 100 steps gave:

```textSpatial Forcing loss slope:     -0.004559 per optimizer stepcosine similarity slope:        +0.004559 per optimizer stepnavigation loss slope:          -0.003115 per optimizer steptotal loss slope:               -0.004483 per optimizer step```

The identity between the alignment-loss decrease and cosine increase is expected because the logged loss is exactly `1 - mean_cosine_similarity`.

All 100 distributed metric records reported:

```textprojector_grad_verified = 1student_grad_verified   = 1vggt_has_gradient       = 0current Qwen tokens     = 192resized teacher tokens  = 192raw VGGT tokens         = 768```

The final log scan found no traceback, OSError, syntax error, NaN, or Inf.

## 6. Verification performed

### Unit tests

```text10 passed in 84.71 s```

Coverage includes explicit final-image masks, wrong-count rejection, bounded JSON-array reads, root-relative paths, 2-D teacher resizing, cosine-loss correctness, student/projector gradients, and teacher detachment.

### Real JanusVLN data validation

The validator ran on source records at indices 0, 1, 3, and 8:

| Frames | Current index | Current grid | Current tokens ||---:|---:|---:|---:|| 1 | 0 | `[1, 24, 32]` | 192 || 2 | 1 | `[1, 24, 32]` | 192 || 4 | 3 | `[1, 24, 32]` | 192 || 9 | 8 | `[1, 24, 32]` | 192 |

The final compute-node invocation exited 0. A prior login-node-only invocation exhausted that host's Rayon thread quota; setting `TOKENIZERS_PARALLELISM=false` and `RAYON_NUM_THREADS=1` or running inside Slurm resolves this environment limit. Distributed training already sets tokenizer parallelism off and was unaffected.

### Full one-GPU forward/backward

A real Qwen3.5-4B + VGGT-1B pass on an R2R record produced:

```textnavigation_loss             1.6316921spatial_forcing_loss        0.9930190total_loss                  1.9295977mean_cosine_similarity      0.0069810current_qwen_token_count    192teacher_token_count         192teacher_raw_token_count     768projector_gradient_norm     0.2600054projector gradient          verifiedQwen student gradient       verifiedVGGT gradient               absentmax CUDA allocation         22.34 GiB```

This test isolated the actual loss graph before DeepSpeed optimizer state allocation.

### Four-H100 training

The final 100-step run exited 0 and saved a 9.7-GiB Hugging Face checkpoint at:

```text/scratch/11528/anhdao69/spatialstack_runs/sf_r2r_smoke_final_4h100_20260801```

The complete per-step metrics are in `train.log` and `trainer_state.json`; `/usr/bin/time -v` output is in `time.txt`.

### Teacher-free stock-Qwen evaluation

`validate_spatial_forcing_inference.py` loaded the saved directory with stock `Qwen3_5ForConditionalGeneration`, then evaluated a real R2R sample:

```json{"model_class": "Qwen3_5ForConditionalGeneration","forbidden_modules_initialized": [],"finite_navigation_loss": 0.2046364993,"vggt_initialized": false,"projector_initialized": false,"geometry_fusion_initialized": false}```

Stock Transformers reports the six saved projector tensors as expected/ignored keys; it does not instantiate or execute them.

## 7. Reproduction

From the repository root:

```bashuv pip install --python .venv/bin/python --no-deps triton==3.7.1

MODEL_PATH=/path/to/Qwen3.5-4B \TEACHER_MODEL_PATH=/path/to/VGGT-1B \JANUSVLN_DATA_ROOT=/scratch/11528/anhdao69/data/JanusVLN_data \OUTPUT_DIR=/path/to/output \NPROC_PER_NODE=4 \bash scripts/train/train_spatial_forcing_vln.sh```

For a bounded smoke test, additionally set a positive `MAX_STEPS`, `MAX_SAMPLES`, and optionally `SAVE_STRATEGY=no`. For full training, use `train_spatial_forcing_vln_full.sh`; it unsets `MAX_STEPS`, consumes all samples, and runs one epoch. The default launcher otherwise matches the plan: batch 1/GPU, accumulation 8, bf16, gradient checkpointing, ZeRO-2, Qwen/merger trainable, Qwen vision frozen, frozen current-only VGGT, layer 24/23 alignment, and weight 0.3.

## 8. Scope and remaining evaluation

This verifies implementation correctness, gradient routing, distributed trainability, teacher-free inference, and the expected optimization direction. It is not a full R2R navigation-quality experiment: no complete-epoch checkpoint or simulator metrics such as SR, SPL, or NE were requested or produced. The next scientific step is a full one-epoch R2R run followed by the standard JanusVLN navigation evaluator, then an ablation against the same Qwen setup with `sf_loss_weight=0`.

## 9. Public-release cleanup and revalidation

Before publication, the repository was narrowed to the code exercised by this project. Legacy `lmms_eval`, Qwen2/Qwen2.5, Pi3, SpatialStack geometry-fusion, benchmark, inference-demo, paper, plan, and local reference-repository trees are not part of the release snapshot. The JanusVLN and Spatial-Forcing repositories remain available locally for provenance but are explicitly ignored and were not staged. VGGT camera, depth, point, and tracking heads were changed to lazy imports; the Spatial Forcing teacher constructs none of them and retains the same aggregator checkpoint-loading path.

The cleaned Qwen3.5-only training entry point was revalidated on the same active 4-H100 Slurm allocation:

```textoutput:                         /scratch/11528/anhdao69/spatialstack_runs/sf_release_clean_4h100_20260801world size:                     4R2R records:                    256optimizer steps:                5trainer time:                   70.07 sSlurm step elapsed:             2 min 52 sSlurm state / exit code:        COMPLETED / 0:0total loss, step 1 -> 5:        2.075 -> 0.6871navigation loss, step 1 -> 5:   1.769 -> 0.3831alignment loss, step 1 -> 5:    1.021 -> 1.013cosine, step 1 -> 5:            -0.02086 -> -0.01331```

All five records again reported 192 Qwen tokens, 192 resized teacher tokens, 768 raw VGGT tokens, nonzero projector/student gradients, and zero VGGT gradients. The first step included one-time Hopper kernel compilation; subsequent optimizer steps took approximately 5-6 seconds. A post-cleanup real-data validator also exited 0 for 1/2/4/9-frame R2R examples, and the focused suite completed with `10 passed` in the requested project `.venv`.

Finally, the exact 38-file root commit was exported to an isolated directory and tested independently of the larger working tree. Its unit tests and wheel-content audit passed, and a two-step 4-H100 launch from that archive completed with Slurm exit code `0:0`; total loss decreased from 2.329 to 2.151, all token/gradient invariants passed, and final checkpoint serialization succeeded.

## 10. Optional explicit VGGT positional embedding

An opt-in `sf_use_vggt_pe` path was subsequently added to reproduce the optional `use_vggt_pe` preprocessing from the Spatial-Forcing OpenVLA reference. This is separate from VGGT's existing learned DINOv2 position embedding and aggregator 2-D RoPE. When enabled, it:

1. constructs a normalized UV grid at the raw VGGT patch resolution;2. scales horizontal and vertical spans using the current image aspect ratio;3. maps each coordinate to a 2-D sine/cosine embedding with `omega_0=100`;4. multiplies the embedding by the reference scale `0.1`;5. adds it to the frozen VGGT feature grid in float32 before bilinear resizing.

The public controls are:

```textmodel argument:  --sf_use_vggt_pe True|Falselauncher env:    SF_USE_VGGT_PE=True|Falsedefault:         False```

The false setting preserves the previously verified behavior. The true setting is saved in the student config but creates no parameter and no inference module. The implementation does not import VGGT depth/tracking heads merely to obtain their position utilities.

Verification with the flag enabled included 13 focused tests, a real one-H100 R2R forward/backward, and a 20-step four-H100 optimization run:

```textoutput:                         /scratch/11528/anhdao69/spatialstack_runs/sf_vggt_pe_true_20step_4h100_20260801Slurm state / exit code:        COMPLETED / 0:0trainer time:                   116.9 salignment loss, step 1 -> 20:   1.019 -> 0.9241cosine, step 1 -> 20:           -0.01879 -> 0.07587navigation loss, step 1 -> 20:  1.915 -> 0.2311total loss, step 1 -> 20:       2.220 -> 0.5084```

Every logged step reported `vggt_pos_embed_enabled=1`, 192 Qwen positions, 192 resized teacher positions, 768 raw VGGT patches, student/projector gradients present, and VGGT gradients absent. The resulting checkpoint also passed stock-Qwen inference validation with finite R2R navigation loss and no teacher, projector, or geometry-fusion module initialized.

## 11. Full-dataset configuration

The hard-coded `janusvln_data_root/train_r2r.json` binding was replaced with a SpatialStack-style JSON dataset configuration. Each entry contains `annotation_path`, `data_path`, `dataset_name`, `tag`, and `dataset_format`. A single object loads one annotation; a list supports named selection and the existing `%N` sampling suffix. The root-only R2R interface remains available for compatibility.

The production config points to:

```textannotation_path: /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.jsondata_path:       /mnt/data/vmo-ai-task/anhdh35/JanusVLN```

`scripts/train/train_spatial_forcing_vln_full.sh` fixes the full recipe to one epoch, `MAX_SAMPLES=-1`, `SF_USE_VGGT_PE=True`, 3% warmup, and step-based checkpointing. It explicitly unsets `MAX_STEPS`, so Transformers derives the update count from the dataloader and epoch count. The base launcher only emits `--max_steps` when a positive smoke-test value is supplied.

Checkpoint cadence and retention are exposed as `SAVE_STRATEGY`, `SAVE_STEPS`, and `SAVE_TOTAL_LIMIT`. The full defaults are `steps`, `1000`, and `10`, following SpatialStack's original 1,000-step cadence and 10-checkpoint retention. `SAVE_TOTAL_LIMIT=0` disables checkpoint rotation. The final model is always saved directly in the output root after training, independently of periodic numbered checkpoints. For the verified 19,727-step R2R epoch, 19 periodic saves are triggered, the latest 10 remain under the default retention policy, and the final step-19,727 model is saved at the output root.

The previous R2R annotation was validated through the same new JSON interface. All 631,244 records were scanned: there were zero missing fields, invalid frame counts, or invalid action labels. With four GPUs, batch size 1 per GPU, and accumulation 8, one R2R epoch is 19,727 optimizer steps. Real processor/data-adapter validation passed for 1-, 2-, 4-, and 9-frame samples with 192 current-image tokens in every case. The focused suite passed 16 tests, including config selection, sampling suffixes, and legacy R2R/RxR absolute-path normalization.

A new two-step four-H100 config-path smoke job was submitted as Slurm job `3365524`, but it remained pending with reason `Priority` and no start estimate while the account's unrelated four-GPU job was active. It was canceled before allocation, so this was not a training failure and consumed no GPU time. GPU behavior for the unchanged model/trainer path remains covered by the completed 20-step positional-encoding run documented above; the newly changed dataset-selection path is covered by real R2R adapter validation.