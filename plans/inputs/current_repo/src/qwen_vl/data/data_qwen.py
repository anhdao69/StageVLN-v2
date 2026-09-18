"""JanusVLN R2R data path for plain Qwen3.5 supervised fine-tuning."""

import copy
import itertools
import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"

QWEN3_5_NON_THINKING_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<think>\\n\\n</think>\\n\\n' + message['content'] }}"
    "{% else %}{{ message['content'] }}{% endif %}"
    "{{ '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    "{% endif %}"
)


def read_json_array(path, max_samples=-1):
    """Read all records, or stream only the prefix needed by a smoke run."""
    if max_samples < 0:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    if max_samples == 0:
        return []

    decoder = json.JSONDecoder()
    records, buffer, position = [], "", 0
    with open(path, encoding="utf-8") as handle:
        eof = False
        while len(records) < max_samples:
            while position >= len(buffer) and not eof:
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] in "[,"
            ):
                position += 1
            if position >= len(buffer) or buffer[position] == "]":
                break
            try:
                record, position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            records.append(record)
            if position > 1024 * 1024:
                buffer, position = buffer[position:], 0
    return records


def load_dataset_config(path):
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("dataset_config must contain one dataset object")
    required = {
        "dataset_name",
        "annotation_path",
        "data_path",
        "tag",
        "dataset_format",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"dataset_config is missing: {', '.join(sorted(missing))}")
    if config["dataset_format"] != "janusvln":
        raise ValueError("Only dataset_format='janusvln' is supported")
    config["annotation_path"] = os.path.expandvars(
        os.path.expanduser(config["annotation_path"])
    )
    config["data_path"] = os.path.expandvars(os.path.expanduser(config["data_path"]))
    return config


def normalize_image_path(image_path, dataset_root):
    root = Path(dataset_root).expanduser().resolve()
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        relative = path
    else:
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            marker = next(
                (part for part in ("R2R-CE-640x480", "R2R") if part in path.parts),
                None,
            )
            if marker is None:
                raise ValueError(f"Image path is outside the dataset root: {path}")
            relative = Path(*path.parts[path.parts.index(marker) :])
    if ".." in relative.parts:
        raise ValueError(f"Image path escapes the dataset root: {image_path}")
    return relative.as_posix()


def _apply_chat_template(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def _assistant_prefix_length(tokenizer):
    empty = _apply_chat_template(tokenizer, [{"role": "assistant", "content": ""}])
    closing = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    return max(0, len(empty) - len(closing))


def tokenize_conversation(conversations, tokenizer, grid_token_counts):
    roles = {"human": "user", "gpt": "assistant"}
    input_ids = _apply_chat_template(
        tokenizer,
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    labels = [IGNORE_INDEX] * len(input_ids)
    assistant_prefix = _assistant_prefix_length(tokenizer)
    grid_index = 0

    for message in conversations:
        role = roles.get(message.get("from", message.get("role")))
        content = message.get("value", message.get("content"))
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Invalid conversation message: {message}")
        if role == "user" and IMAGE_TOKEN in content:
            pieces = content.split(IMAGE_TOKEN)
            rebuilt = []
            for piece in pieces[:-1]:
                if grid_index >= len(grid_token_counts):
                    raise ValueError("Conversation has more <image> tags than images")
                rebuilt.extend(
                    (
                        piece,
                        "<|vision_start|>",
                        "<|image_pad|>" * grid_token_counts[grid_index],
                        "<|vision_end|>",
                    )
                )
                grid_index += 1
            rebuilt.append(pieces[-1])
            content = "".join(rebuilt)

        encoded = _apply_chat_template(
            tokenizer, [{"role": role, "content": content}]
        )
        input_ids.extend(encoded)
        if role == "assistant":
            target = encoded.copy()
            target[:assistant_prefix] = [IGNORE_INDEX] * min(
                assistant_prefix, len(target)
            )
            labels.extend(target)
        else:
            labels.extend([IGNORE_INDEX] * len(encoded))

    if grid_index != len(grid_token_counts):
        raise ValueError(
            f"Conversation has {grid_index} <image> tags for "
            f"{len(grid_token_counts)} images"
        )
    return torch.tensor(input_ids), torch.tensor(labels)


class R2RSFTDataset(Dataset):
    def __init__(self, tokenizer, data_args):
        self.config = load_dataset_config(data_args.dataset_config)
        self.data_root = self.config["data_path"]
        records = read_json_array(
            self.config["annotation_path"], data_args.max_samples
        )
        self.records = []
        allowed_actions = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}
        for record in records:
            images = record.get("images")
            if not isinstance(images, list) or not images:
                raise ValueError("Every JanusVLN record needs an ordered images list")
            if len(images) > data_args.max_history_frames + 1:
                raise ValueError(
                    f"Record has {len(images) - 1} history frames; "
                    f"the configured maximum is {data_args.max_history_frames}"
                )
            record["images"] = [
                normalize_image_path(path, self.data_root) for path in images
            ]
            conversations = record.get("conversations")
            if not conversations:
                raise ValueError("Every JanusVLN record needs conversations")
            action = conversations[-1].get("value", conversations[-1].get("content"))
            if action not in allowed_actions:
                raise ValueError(f"Invalid R2R action: {action!r}")
            self.records.append(record)

        if data_args.shuffle:
            random.shuffle(self.records)
        if not self.records:
            raise ValueError("The training dataset is empty")

        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer.chat_template = QWEN3_5_NON_THINKING_CHAT_TEMPLATE
        self.image_processor = data_args.processor.image_processor
        self.image_processor.max_pixels = data_args.max_pixels
        self.image_processor.min_pixels = data_args.min_pixels
        self.image_processor.size["longest_edge"] = data_args.max_pixels
        self.image_processor.size["shortest_edge"] = data_args.min_pixels
        print(f">>>>> loaded {len(self.records):,} R2R samples")

    def __len__(self):
        return len(self.records)

    @property
    def modality_lengths(self):
        return [
            len(record["images"]) * 252
            + sum(
                len(message.get("value", message.get("content", "")).split())
                for message in record["conversations"]
            )
            for record in self.records
        ]

    def __getitem__(self, index):
        record = self.records[index]
        images = []
        for relative_path in record["images"]:
            image_path = os.path.join(self.data_root, relative_path)
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        processed = self.image_processor.preprocess(images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        grids = list(processed["image_grid_thw"])

        merge_size = self.image_processor.merge_size
        token_counts = [int(grid.prod().item()) // merge_size**2 for grid in grids]
        input_ids, labels = tokenize_conversation(
            record["conversations"], self.tokenizer, token_counts
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grids,
        }


@dataclass
class DataCollatorForSFT:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        max_length = self.tokenizer.model_max_length
        input_ids, labels = input_ids[:, :max_length], labels[:, :max_length]
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        mm_token_type_ids[input_ids == image_token_id] = 1

        grids = torch.stack(
            list(
                itertools.chain.from_iterable(
                    item["image_grid_thw"] for item in instances
                )
            )
        )
        expected_tokens = int((grids.prod(dim=-1) // 4).sum().item())
        actual_tokens = int((input_ids == image_token_id).sum().item())
        if actual_tokens != expected_tokens:
            raise ValueError(
                "Image tokens were truncated or incorrectly encoded: "
                f"actual={actual_tokens}, expected={expected_tokens}"
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": torch.cat(
                [item["pixel_values"] for item in instances]
            ),
            "image_grid_thw": grids,
        }


def make_supervised_data_module(tokenizer, data_args):
    dataset = R2RSFTDataset(tokenizer, data_args)
    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": DataCollatorForSFT(tokenizer),
    }
