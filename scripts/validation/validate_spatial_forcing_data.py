#!/usr/bin/env python
"""Validate JanusVLN frame ordering and Qwen current-image masks."""

import argparse
import json
import os
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

from transformers import AutoProcessor, AutoTokenizer

from qwen_vl.data.data_qwen import make_supervised_data_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        padding_side="right",
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
        max_samples=9,
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
    module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = module["train_dataset"]
    collator = module["data_collator"]

    validation = []
    for dataset_index, expected_frames in ((0, 1), (1, 2), (3, 4), (8, 9)):
        sample = dataset[dataset_index]
        batch = collator([sample])
        token_count = int(batch["current_image_token_mask"].sum().item())
        grid = batch["current_image_grid_thw"][0].tolist()
        expected_tokens = (
            int(grid[0])
            * (int(grid[1]) // collator.spatial_merge_size)
            * (int(grid[2]) // collator.spatial_merge_size)
        )
        assert int(sample["frame_count"].item()) == expected_frames
        assert int(sample["current_frame_index"].item()) == expected_frames - 1
        assert token_count == expected_tokens
        assert sample["ordered_frame_paths"][-1].endswith(
            dataset.list_data_dict[dataset_index]["id"]
        )
        validation.append(
            {
                "dataset_index": dataset_index,
                "frame_count": expected_frames,
                "current_frame_index": expected_frames - 1,
                "current_grid_thw": grid,
                "current_token_count": token_count,
            }
        )

    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
