#!/usr/bin/env python
"""Validate v4 depth supervision on real JanusVLN records.

This utility runs the real one-Qwen/one-VGGT training path without backward,
checks the enforced invariants, records depth distributions, and renders RGB /
VGGT pseudo-depth / student depth / absolute-error panels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from qwen_vl.data.data_qwen import make_supervised_data_module
from qwen_vl.model.depth_supervision import finite_depth_statistics
from qwen_vl.model.spatial_forcing import (
    Qwen3_5ForConditionalGenerationWithSpatialForcing,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--dataset-use", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--visualizations", type=int, default=4)
    return parser.parse_args()


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list) and all(
            isinstance(item, torch.Tensor) for item in value
        ):
            moved[key] = [item.to(device) for item in value]
        else:
            moved[key] = value
    return moved


def colorize(values, *, minimum=None, maximum=None):
    values = values.detach().float().cpu().numpy()
    finite = np.isfinite(values)
    if minimum is None:
        minimum = float(np.nanpercentile(values[finite], 1)) if finite.any() else 0.0
    if maximum is None:
        maximum = float(np.nanpercentile(values[finite], 99)) if finite.any() else 1.0
    scale = max(maximum - minimum, 1e-8)
    normalized = np.clip((values - minimum) / scale, 0.0, 1.0)
    normalized[~finite] = 0.0
    # Compact blue-cyan-yellow diagnostic map without an extra plotting dependency.
    red = np.clip(2.0 * normalized - 0.25, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.25 - 2.0 * normalized, 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def render_panel(path, rgb, teacher, prediction):
    rgb = (
        rgb.detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255
    ).astype(np.uint8)
    finite_teacher = teacher[torch.isfinite(teacher)]
    if finite_teacher.numel():
        minimum = float(torch.quantile(finite_teacher.float(), 0.01).item())
        maximum = float(torch.quantile(finite_teacher.float(), 0.99).item())
    else:
        minimum, maximum = 0.0, 1.0
    images = [
        Image.fromarray(rgb),
        Image.fromarray(colorize(teacher, minimum=minimum, maximum=maximum)),
        Image.fromarray(colorize(prediction, minimum=minimum, maximum=maximum)),
        Image.fromarray(colorize((prediction - teacher).abs())),
    ]
    labels = ["Current RGB", "VGGT pseudo-depth", "Student depth", "Absolute error"]
    width, height = images[0].size
    canvas = Image.new("RGB", (4 * width, height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        canvas.paste(image, (index * width, 24))
        draw.text((index * width + 4, 5), label, fill="black")
    canvas.save(path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    config.use_geometry_encoder = False
    config.use_geometry_fusion = False
    config.spatial_forcing_enabled = True
    config.sf_loss_weight = 0.3
    config.sf_student_layer = 24
    config.sf_teacher_layer = 23
    config.sf_use_vggt_pe = True
    config.sf_projector_hidden_dim = 4096
    config.sf_teacher_dim = 2048
    config.sf_verify_invariants = True
    config.sf_multiframe_teacher = False
    config.depth_supervision_enabled = True
    config.depth_loss_weight = 0.05
    config.depth_student_layers = [7, 16, 24, 32]
    config.depth_loss_type = "geo_depth"
    config.depth_gradient_scales = [1, 2, 4, 8]
    config.depth_outlier_keep_ratio = 0.98
    config.depth_use_teacher_confidence = False
    config.use_cache = False
    config.text_config.use_cache = False
    qwen_vision_patch_size = int(config.vision_config.patch_size)

    model = Qwen3_5ForConditionalGenerationWithSpatialForcing.from_pretrained(
        args.model_path,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        spatial_teacher_model_path=args.teacher_path,
    ).cuda()
    model.model.visual.requires_grad_(False)
    model.train()

    processor = AutoProcessor.from_pretrained(
        args.model_path, cache_dir=args.cache_dir, padding_side="right"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        model_max_length=12800,
        padding_side="right",
        use_fast=False,
    )
    data_args = SimpleNamespace(
        dataset_use=args.dataset_use,
        dataset_config=args.dataset_config,
        janusvln_data_root=None,
        max_samples=args.start_index + args.max_samples,
        shuffle=False,
        model_type="qwen3.5",
        spatial_forcing_enabled=True,
        depth_supervision_enabled=True,
        use_geometry_encoder=False,
        image_processor=processor.image_processor,
        max_pixels=576 * 28 * 28,
        min_pixels=16 * 28 * 28,
        video_max_frames=8,
        video_min_frames=4,
        data_flatten=False,
        base_interval=2,
        video_max_frame_pixels=1664 * 28 * 28,
        video_min_frame_pixels=256 * 28 * 28,
    )
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]

    captured = {}
    aggregator_calls = 0

    def count_aggregator(_module, _inputs, _output):
        nonlocal aggregator_calls
        aggregator_calls += 1

    def capture_teacher_depth(_module, _inputs, output):
        captured["teacher"] = output[0].detach()

    def capture_student_depth(_module, _inputs, output):
        captured["student"] = output.detach()

    handles = [
        model.spatial_teacher.vggt.aggregator.register_forward_hook(count_aggregator),
        model.spatial_teacher.vggt.depth_head.register_forward_hook(
            capture_teacher_depth
        ),
        model.student_depth_head.register_forward_hook(capture_student_depth),
    ]

    records = []
    try:
        stop_index = min(args.start_index + args.max_samples, len(dataset))
        for sample_index in range(args.start_index, stop_index):
            captured.clear()
            calls_before = aggregator_calls
            batch = collator([dataset[sample_index]])
            rgb = batch["sf_teacher_pixel_values"][0]
            grid = batch["current_image_grid_thw"][0]
            expected_hw = (
                int(grid[1]) * qwen_vision_patch_size,
                int(grid[2]) * qwen_vision_patch_size,
            )
            if tuple(rgb.shape[-2:]) != expected_hw:
                raise AssertionError(
                    f"Aligned VGGT input {tuple(rgb.shape[-2:])} != Qwen grid {expected_hw}"
                )
            batch = move_batch_to_device(batch, torch.device("cuda"))
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
            calls_this_sample = aggregator_calls - calls_before
            if calls_this_sample != 1:
                raise AssertionError(
                    f"Expected one VGGT aggregator call, observed {calls_this_sample}"
                )
            teacher = captured["teacher"][0, 0, ..., 0]
            student = captured["student"][0]
            teacher = teacher[..., : student.shape[-2], : student.shape[-1]]
            if teacher.shape != student.shape:
                raise AssertionError(
                    f"Depth shape mismatch: {tuple(teacher.shape)} vs {tuple(student.shape)}"
                )
            metrics = {
                name: float(value.detach().float().mean().item())
                for name, value in model.last_spatial_forcing_metrics.items()
            }
            metrics["loss"] = float(outputs.loss.detach().float().item())
            metrics["sample_index"] = sample_index
            metrics["aggregator_calls"] = calls_this_sample
            metrics.update(
                {
                    f"teacher_direct_{name}": float(value.item())
                    for name, value in finite_depth_statistics(teacher).items()
                }
            )
            records.append(metrics)
            if len(records) <= args.visualizations:
                render_panel(
                    output_dir / f"depth_sample_{sample_index:03d}.png",
                    rgb,
                    teacher,
                    student,
                )
    finally:
        for handle in handles:
            handle.remove()

    numeric_keys = sorted(
        set.intersection(
            *(set(record) for record in records)
        )
        - {"sample_index"}
    )
    summary = {
        "dataset_use": args.dataset_use,
        "sample_count": len(records),
        "means": {
            key: float(np.mean([record[key] for record in records]))
            for key in numeric_keys
        },
        "records": records,
    }
    with (output_dir / "depth_validation.json").open("w") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
