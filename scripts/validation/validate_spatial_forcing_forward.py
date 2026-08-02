#!/usr/bin/env python
"""Run one real Qwen3.5 + VGGT Spatial Forcing forward/backward pass."""

import argparse
import json
import os
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from qwen_vl.data.data_qwen import make_supervised_data_module
from qwen_vl.model.spatial_forcing import (
    Qwen3_5ForConditionalGenerationWithSpatialForcing,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", default=None)
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


def main():
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    config.use_geometry_encoder = False
    config.use_geometry_fusion = False
    config.spatial_forcing_enabled = True
    config.sf_loss_weight = 0.3
    config.sf_student_layer = 24
    config.sf_teacher_layer = 23
    config.sf_projector_hidden_dim = 4096
    config.sf_teacher_dim = 2048
    config.sf_verify_invariants = True
    config.use_cache = False

    model = Qwen3_5ForConditionalGenerationWithSpatialForcing.from_pretrained(
        args.model_path,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        spatial_teacher_model_path=args.teacher_path,
    )
    model.model.visual.requires_grad_(False)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train().cuda()

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
        dataset_use="janusvln_r2r",
        janusvln_data_root=args.data_root,
        max_samples=1,
        shuffle=False,
        model_type="qwen3.5",
        spatial_forcing_enabled=True,
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
    batch = move_batch_to_device(batch, torch.device("cuda"))

    outputs = model(**batch)
    forward_metrics = {
        key: float(value.float().mean().item())
        for key, value in model.last_spatial_forcing_metrics.items()
    }
    outputs.loss.backward()
    teacher_has_gradient = any(
        parameter.grad is not None for parameter in model.spatial_teacher.parameters()
    )
    projector_gradient_norm = torch.sqrt(
        sum(
            parameter.grad.float().pow(2).sum()
            for parameter in model.spatial_projector.parameters()
            if parameter.grad is not None
        )
    )
    result = {
        **forward_metrics,
        "projector_gradient_norm": float(projector_gradient_norm.item()),
        "projector_grad_verified": model._sf_projector_grad_verified,
        "student_grad_verified": model._sf_student_grad_verified,
        "vggt_has_gradient": teacher_has_gradient,
        "max_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    print(json.dumps(result, indent=2))
    assert result["projector_gradient_norm"] > 0
    assert result["projector_grad_verified"]
    assert result["student_grad_verified"]
    assert not result["vggt_has_gradient"]


if __name__ == "__main__":
    main()
