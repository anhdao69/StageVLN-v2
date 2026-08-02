import os
import copy
import json
import random
import time
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_35
from .utils import prepare_image_inputs

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

QWEN_TRAIN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

QWEN3_5_NON_THINKING_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}{{ '<think>\n\n</think>\n\n' + message['content'] }}{% else %}{{ message['content'] }}{% endif %}"
    "{{ '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{% endif %}"
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path, max_samples: int = -1):
    with open(path, "r") as f:
        # return [json.loads(line) for line in f]
        ret = []
        for line in f:
            ret.append(json.loads(line))
            if max_samples != -1 and len(ret) >= max_samples:
                break
    return ret


def read_json_array(path: str, max_samples: int = -1):
    """Read a JSON array, without loading a multi-GB file for smoke tests."""
    if max_samples < 0:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if max_samples == 0:
        return []

    decoder = json.JSONDecoder()
    records = []
    buffer = ""
    position = 0
    reached_eof = False

    with open(path, "r", encoding="utf-8") as handle:
        while len(records) < max_samples:
            if position > 1024 * 1024:
                buffer = buffer[position:]
                position = 0

            while position >= len(buffer) and not reached_eof:
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    reached_eof = True

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] in "[,"
            ):
                position += 1

            if position < len(buffer) and buffer[position] == "]":
                break

            try:
                record, end_position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if reached_eof:
                    raise
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    reached_eof = True
                continue

            records.append(record)
            position = end_position

    return records


def normalize_janusvln_image_path(image_path: str, dataset_root: str) -> str:
    """Return a dataset-root-relative JanusVLN image path."""
    root = Path(dataset_root).expanduser().resolve()
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        relative_path = path
    else:
        try:
            relative_path = path.resolve().relative_to(root)
        except ValueError:
            marker = "R2R-CE-640x480"
            if marker not in path.parts:
                raise ValueError(
                    f"JanusVLN image path is outside {root} and has no {marker} component: {path}"
                )
            marker_index = path.parts.index(marker)
            relative_path = Path(*path.parts[marker_index:])

    if ".." in relative_path.parts:
        raise ValueError(f"JanusVLN image path escapes its dataset root: {image_path}")
    return relative_path.as_posix()


def build_current_image_token_mask(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    expected_token_count: int,
) -> torch.Tensor:
    """Select the image-pad tokens belonging to the final/current image."""
    if input_ids.ndim != 1:
        raise ValueError(
            f"input_ids must be one-dimensional, got {tuple(input_ids.shape)}"
        )

    start_positions = torch.nonzero(
        input_ids.eq(vision_start_token_id), as_tuple=False
    ).flatten()
    if start_positions.numel() == 0:
        raise ValueError(
            "No vision-start token found while building current-image mask"
        )

    final_start = int(start_positions[-1].item())
    following_end_positions = torch.nonzero(
        input_ids[final_start + 1 :].eq(vision_end_token_id), as_tuple=False
    ).flatten()
    if following_end_positions.numel() == 0:
        raise ValueError("No vision-end token follows the final vision-start token")
    final_end = final_start + 1 + int(following_end_positions[0].item())

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    current_span = input_ids[final_start + 1 : final_end]
    current_positions = torch.nonzero(
        current_span.eq(image_token_id), as_tuple=False
    ).flatten()
    mask[final_start + 1 + current_positions] = True

    actual_count = int(mask.sum().item())
    if actual_count != int(expected_token_count):
        raise ValueError(
            "Current-image token mask does not match the final Qwen image grid: "
            f"mask={actual_count}, expected={expected_token_count}"
        )
    return mask


def _extract_input_ids(encoded):
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        return list(encoded["input_ids"])
    return list(encoded)


def _build_training_tokenizer(
    tokenizer: transformers.PreTrainedTokenizer,
    model_type: str,
) -> transformers.PreTrainedTokenizer:
    tokenizer = copy.deepcopy(tokenizer)
    if model_type == "qwen3.5":
        tokenizer.chat_template = QWEN3_5_NON_THINKING_CHAT_TEMPLATE
    else:
        tokenizer.chat_template = QWEN_TRAIN_CHAT_TEMPLATE
    return tokenizer


def _apply_training_chat_template(
    tokenizer: transformers.PreTrainedTokenizer,
    messages,
    *,
    add_generation_prompt: bool = False,
):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return _extract_input_ids(encoded)


def _get_assistant_prefix_length(tokenizer: transformers.PreTrainedTokenizer) -> int:
    if hasattr(tokenizer, "_assistant_prefix_length"):
        return tokenizer._assistant_prefix_length

    empty_assistant_ids = _apply_training_chat_template(
        tokenizer,
        [{"role": "assistant", "content": ""}],
    )
    closing_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    assistant_prefix_length = max(0, len(empty_assistant_ids) - len(closing_ids))
    tokenizer._assistant_prefix_length = assistant_prefix_length
    return assistant_prefix_length


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
    model_type: str = "qwen2.5vl",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = _build_training_tokenizer(tokenizer, model_type)
    assistant_prefix_length = _get_assistant_prefix_length(tokenizer)

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except (KeyError, IndexError):
            print(sources)

        input_id, target = [], []

        input_id += _apply_training_chat_template(
            tokenizer,
            [{"role": "system", "content": system_message}],
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except KeyError:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = _apply_training_chat_template(tokenizer, conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:assistant_prefix_length] = [IGNORE_INDEX] * min(
                    assistant_prefix_length, len(target_mask)
                )
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(
            dataset,
            janusvln_data_root=getattr(data_args, "janusvln_data_root", None),
        )
        print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        self.spatial_forcing_enabled = getattr(
            data_args, "spatial_forcing_enabled", False
        )
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen3.5":
            self.get_rope_index = get_rope_index_35
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(
                    data["annotation_path"], max_samples=data_args.max_samples
                )
            else:
                annotations = read_json_array(
                    data["annotation_path"], max_samples=data_args.max_samples
                )
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if data["dataset_name"] == "janusvln_r2r":
                    frame_paths = ann.get("images")
                    if (
                        not isinstance(frame_paths, list)
                        or not 1 <= len(frame_paths) <= 9
                    ):
                        raise ValueError(
                            "JanusVLN R2R records must contain 1-9 ordered images"
                        )
                    ann["images"] = [
                        normalize_janusvln_image_path(path, data["data_path"])
                        for path in frame_paths
                    ]
                    action = ann.get("conversations", [{}, {}])[-1].get("value")
                    if action not in {
                        "MOVE_FORWARD",
                        "TURN_LEFT",
                        "TURN_RIGHT",
                        "STOP",
                    }:
                        raise ValueError(f"Invalid JanusVLN action label: {action}")
                ann["data_path"] = data["data_path"]
                ann["tag"] = data["tag"]
                ann["dataset_name"] = data["dataset_name"]
            list_data_dict += annotations

        print(f"Total training samples: {len(list_data_dict)}")

        if data_args.shuffle:
            random.shuffle(list_data_dict)

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            length_list.append(image_num * 252 + cur_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            cur_len += image_num * 252
            tag = sample.get("tag", "2d")
            cur_len = -cur_len if tag == "2d" else cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def draw_visual_marks(self, images, spar_info):

        if spar_info is None:
            return
        info = json.loads(spar_info)
        task_type = info["type"]
        from .draw_marker import DRAW_FUNCTIONS

        draw_fn = DRAW_FUNCTIONS[task_type]
        if len(images) == 1:
            draw_fn(images[0], info)
        else:
            draw_fn(images, info)
        # for j, img in enumerate(images):
        #     # write to local
        #     img.save(f"images/img_{j}.jpg", format="JPEG")

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def read_video_images(self, source):
        # read video images from the source
        assert isinstance(source["video"], str), "video should be a string"
        video_file = os.path.join(source["data_path"], source["video"])
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
            raise FileNotFoundError

        def get_frame_indices(total_frames, fps=1):
            video_length = total_frames / fps
            interval = getattr(self.data_args, "base_interval", 2)
            num_frames_to_sample = round(video_length / interval)
            video_min_frames = getattr(self.data_args, "video_min_frames", 4)
            video_max_frames = getattr(self.data_args, "video_max_frames", 8)
            target_frames = min(
                max(num_frames_to_sample, video_min_frames), video_max_frames
            )
            frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            frame_idx = np.unique(frame_idx)
            return frame_idx

        # check whether video_file is a directory
        if os.path.isdir(video_file):
            frame_files = [
                os.path.join(video_file, f)
                for f in os.listdir(video_file)
                if os.path.isfile(os.path.join(video_file, f))
            ]
            frame_files.sort()
            frame_idx = get_frame_indices(len(frame_files), 1)
            images = [frame_files[i] for i in frame_idx]
            images = [Image.open(frame).convert("RGB") for frame in images]
        elif any([video_file.endswith(ext) for ext in [".mp4", ".avi", ".mov"]]):
            vr = VideoReader(video_file, num_threads=4)
            total_frames = len(vr)
            avg_fps = vr.get_avg_fps()
            frame_idx = get_frame_indices(total_frames, avg_fps)
            video = vr.get_batch(frame_idx).asnumpy()

            images = [Image.fromarray(frame).convert("RGB") for frame in video]
        return images

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        source_record = copy.deepcopy(self.list_data_dict[i])
        sources = [source_record]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video = None

        if "video" in sources[0]:
            sources[0]["images"] = self.read_video_images(sources[0])
            num_image = len(sources[0]["images"])
            conv_value = sources[0]["conversations"][0]["value"]
            replacement_tokens = "".join([DEFAULT_IMAGE_TOKEN] * num_image)
            if DEFAULT_VIDEO_TOKEN in conv_value:
                conv_value = conv_value.replace(DEFAULT_VIDEO_TOKEN, replacement_tokens)
            elif DEFAULT_IMAGE_TOKEN in conv_value:
                conv_value = conv_value.replace(DEFAULT_IMAGE_TOKEN, replacement_tokens)
            else:
                conv_value = replacement_tokens + conv_value

            sources[0]["conversations"][0]["value"] = conv_value

            del sources[0]["video"]

        # # replace <image>\n with <image>
        sources[0]["conversations"][0]["value"] = sources[0]["conversations"][0][
            "value"
        ].replace(f"{DEFAULT_IMAGE_TOKEN}\n", DEFAULT_IMAGE_TOKEN)

        # rename images tag
        if "images" in sources[0]:
            sources[0]["image"] = sources[0]["images"]

        # notice that we use images as the tag
        if "image" in sources[0]:
            image_folder = source_record["data_path"]
            image_file = source_record["image"]
            ordered_frame_paths = tuple(
                str(frame) for frame in image_file if isinstance(frame, str)
            )
            if isinstance(image_file, list):
                if isinstance(image_file[0], str):
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    image_file = [Image.open(img).convert("RGB") for img in image_file]
                elif isinstance(image_file[0], Image.Image):
                    pass
                else:
                    raise NotImplementedError
                # draw visual markers
                self.draw_visual_marks(image_file, sources[0].get("spar_info", None))

                image, grid_thw, geometry_encoder_inputs = [], [], []
                sf_teacher_pixel_values = None
                use_geometry_encoder = getattr(
                    self.data_args, "use_geometry_encoder", False
                )
                for frame_index, file in enumerate(image_file):
                    is_current_frame = frame_index == len(image_file) - 1
                    prepare_geometry = use_geometry_encoder or (
                        self.spatial_forcing_enabled and is_current_frame
                    )
                    ret = prepare_image_inputs(
                        file,
                        self.data_args.image_processor,
                        model_type=self.model_type,
                        prepare_geometry=prepare_geometry,
                    )
                    image.append(ret["pixel_values"])
                    grid_thw.append(ret["image_grid_thw"])
                    if use_geometry_encoder:
                        geometry_encoder_inputs.append(ret["geometry_encoder_inputs"])
                    if self.spatial_forcing_enabled and is_current_frame:
                        sf_teacher_pixel_values = ret["geometry_encoder_inputs"]
            else:
                raise NotImplementedError

            grid_thw_merged = copy.deepcopy(grid_thw)
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources,
                self.tokenizer,
                grid_thw=grid_thw_merged,
                visual_type="image",
                model_type=self.model_type,
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )
        elif "video" in sources[0]:
            video_file = source_record["video"]
            video_folder = source_record["data_path"]
            if isinstance(video_file, list):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources,
                self.tokenizer,
                grid_thw=grid_thw_merged,
                visual_type="video",
                model_type=self.model_type,
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources,
                self.tokenizer,
                grid_thw=grid_thw_merged,
                model_type=self.model_type,
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )

        if "image" in source_record:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
            if getattr(self.data_args, "use_geometry_encoder", False):
                data_dict["geometry_encoder_inputs"] = geometry_encoder_inputs
            if self.spatial_forcing_enabled:
                if source_record.get("dataset_name") != "janusvln_r2r":
                    raise ValueError(
                        "Spatial Forcing currently requires the janusvln_r2r adapter"
                    )
                if sf_teacher_pixel_values is None:
                    raise ValueError("The current VGGT frame tensor was not prepared")

                current_grid_thw = grid_thw[-1]
                expected_current_tokens = int(grid_thw_merged[-1].item())
                image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                vision_start_token_id = self.tokenizer.convert_tokens_to_ids(
                    "<|vision_start|>"
                )
                vision_end_token_id = self.tokenizer.convert_tokens_to_ids(
                    "<|vision_end|>"
                )
                current_image_token_mask = build_current_image_token_mask(
                    data_dict["input_ids"],
                    image_token_id=image_token_id,
                    vision_start_token_id=vision_start_token_id,
                    vision_end_token_id=vision_end_token_id,
                    expected_token_count=expected_current_tokens,
                )
                frame_count = len(image_file)
                data_dict.update(
                    current_image_token_mask=current_image_token_mask,
                    current_image_grid_thw=current_grid_thw,
                    sf_teacher_pixel_values=sf_teacher_pixel_values,
                    ordered_frame_paths=ordered_frame_paths,
                    frame_count=torch.tensor(frame_count, dtype=torch.long),
                    current_frame_index=torch.tensor(frame_count - 1, dtype=torch.long),
                )
        # video exist in the data
        elif "video" in source_record:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw

        data_dict["tag"] = source_record.get("tag", "2d")
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    spatial_merge_size: int = 2

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        has_spatial_forcing = [
            "current_image_token_mask" in instance for instance in instances
        ]
        if any(has_spatial_forcing) and not all(has_spatial_forcing):
            raise ValueError(
                "A batch cannot mix Spatial Forcing and non-Spatial-Forcing records"
            )
        if all(has_spatial_forcing):
            current_image_token_mask = torch.nn.utils.rnn.pad_sequence(
                [instance["current_image_token_mask"] for instance in instances],
                batch_first=True,
                padding_value=False,
            )[:, : self.tokenizer.model_max_length]
            current_image_grid_thw = torch.stack(
                [instance["current_image_grid_thw"] for instance in instances]
            )
            merge_size = int(self.spatial_merge_size)
            expected_counts = current_image_grid_thw.prod(dim=-1) // (merge_size**2)
            actual_counts = current_image_token_mask.sum(dim=-1)
            if not torch.equal(actual_counts.cpu(), expected_counts.cpu()):
                raise ValueError(
                    "Current image tokens were truncated or incorrectly collated: "
                    f"actual={actual_counts.tolist()}, expected={expected_counts.tolist()}"
                )

            frame_count = torch.stack(
                [instance["frame_count"] for instance in instances]
            )
            current_frame_index = torch.stack(
                [instance["current_frame_index"] for instance in instances]
            )
            if not torch.equal(current_frame_index, frame_count - 1):
                raise ValueError("The current JanusVLN frame must be final")

            batch.update(
                current_image_token_mask=current_image_token_mask,
                current_image_grid_thw=current_image_grid_thw,
                sf_teacher_pixel_values=[
                    instance["sf_teacher_pixel_values"] for instance in instances
                ],
                frame_count=frame_count,
                current_frame_index=current_frame_index,
            )

        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            geometry_encoder_inputs = [
                torch.stack(instance["geometry_encoder_inputs"])
                for instance in instances
            ]
            batch["geometry_encoder_inputs"] = geometry_encoder_inputs
            tags = [instance.get("tag", "3d") for instance in instances]
            assert len(set(tags)) == 1, "all data in a batch should have the same tag"
            batch["tag"] = tags[0]
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor(
            [0] + [len(seq) for seq in input_ids], dtype=torch.int32
        )
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            raise NotImplementedError(
                "FlattenedDataCollatorForSupervisedDataset does not support geometry_encoder_inputs"
            )

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            spatial_merge_size=data_args.image_processor.merge_size,
        )
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        spatial_merge_size=data_args.image_processor.merge_size,
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
