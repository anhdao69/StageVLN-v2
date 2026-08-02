#!/usr/bin/env python
"""Verify that a Spatial Forcing checkpoint evaluates with stock Qwen only."""

import argparse
import json
import os
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
)

from qwen_vl.data.data_qwen import make_supervised_data_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            args.checkpoint,
            dtype=torch.bfloat16,
        )
        .eval()
        .cuda()
    )
    forbidden_modules = (
        "spatial_teacher",
        "spatial_projector",
        "geometry_encoder",
        "geometry_merger",
    )
    initialized_forbidden_modules = [
        name for name in forbidden_modules if hasattr(model, name)
    ]
    assert not initialized_forbidden_modules

    processor = AutoProcessor.from_pretrained(args.checkpoint, padding_side="right")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        model_max_length=12800,
        padding_side="right",
        use_fast=False,
    )
    data_args = SimpleNamespace(
        dataset_use="janusvln_r2r",
        janusvln_data_root=args.data_root,
        max_samples=1,
        shuffle=False,
        model_type="qwen3.5",
        spatial_forcing_enabled=False,
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
    batch = data_module["data_collator"]([data_module["train_dataset"][0]])
    batch = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        outputs = model(**batch)
    assert outputs.loss is not None and torch.isfinite(outputs.loss)

    print(
        json.dumps(
            {
                "model_class": type(model).__name__,
                "forbidden_modules_initialized": initialized_forbidden_modules,
                "finite_navigation_loss": float(outputs.loss.item()),
                "vggt_initialized": False,
                "projector_initialized": False,
                "geometry_fusion_initialized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
