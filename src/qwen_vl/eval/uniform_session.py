"""Fixed-weight, memory-free actor for explicit-history R2R exports."""

from contextlib import nullcontext
from dataclasses import replace
import hashlib

import numpy as np
from PIL import Image
import torch

from qwen_vl.contracts import EpisodeRecord, FrameKey, FrameRecord
from qwen_vl.data.history import history_indices
from qwen_vl.data.prompting import build_state
from qwen_vl.eval.session import InvalidActionError, decode_action, greedy_generate
from qwen_vl.models.qwen_adapter import QwenReaderAdapter, QwenVisualAdapter


class UniformHistorySession:
    """Encode each observation once and select the configured history protocol.

    Native merged features are retained on CPU. Uniform4 and Uniform8 retain
    the complete episode prefix; Sliding Window 4 keeps only frames needed by
    the next step.
    No memory writer, learned slots, or Qwen cache persists between decisions.
    """

    def __init__(self, backbone, processor, max_new_tokens=32, history_mode="uniform4"):
        if history_mode not in ("uniform4", "uniform8", "recent"):
            raise ValueError("History mode must be uniform4, uniform8, or recent")
        self.backbone = backbone.eval()
        self.processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.visual = QwenVisualAdapter(backbone)
        self.reader = QwenReaderAdapter(backbone, self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.history_mode = history_mode
        self.episode_id = None
        self.instruction = None
        self.next_step = 0
        self._features = {}
        self._decisions = {}
        self.last_state = self.last_token_ids = None

    @property
    def device(self):
        return self.backbone.get_input_embeddings().weight.device

    @torch.inference_mode()
    def reset(self, episode_id, instruction):
        episode_id = str(episode_id)
        if not episode_id or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Episode identity and instruction must be nonempty")
        if self.episode_id == episode_id and self.instruction != instruction:
            raise ValueError("Conflicting instruction for the same episode")
        self.episode_id, self.instruction = episode_id, instruction
        self.next_step = 0
        self._features = {}
        self._decisions = {}
        self.last_state = self.last_token_ids = None

    @torch.inference_mode()
    def observe(self, step_id, rgb):
        if self.episode_id is None:
            raise ValueError("reset must establish an episode before observing")
        if type(step_id) is not int or step_id < 0:
            raise ValueError("Observation step must be a nonnegative integer")
        if isinstance(rgb, Image.Image):
            image = rgb.convert("RGB")
        elif isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3:
            image = Image.fromarray(rgb)
        else:
            raise TypeError("RGB must be a PIL image or uint8 HWC array")
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if step_id in self._decisions:
            old_digest, result = self._decisions[step_id]
            if old_digest != digest:
                raise ValueError("Observation RGB conflicts with the previously observed step")
            if isinstance(result, InvalidActionError):
                raise InvalidActionError(str(result))
            return result
        if step_id != self.next_step:
            raise ValueError("New observations must be chronological without skipped steps")

        key = FrameKey("r2r", self.episode_id, step_id)
        processed = self.processor.preprocess([image], return_tensors="pt")
        pixels = processed["pixel_values"].to(self.device)
        grid = processed["image_grid_thw"].to(self.device)
        compute = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with compute:
            feature = self.visual.encode(pixels, grid, [key])[0]
            # The writer-only pre-merger tensor is not used by this policy.
            feature = replace(feature, premerge=torch.empty(0), merged=feature.merged.detach().cpu())
            self._features[key] = feature
            selected = history_indices(step_id, self.history_mode, recent=4)
            frames = tuple(FrameRecord(FrameKey("r2r", self.episode_id, index), "", None, "")
                           for index in selected)
            episode = EpisodeRecord(self.episode_id, self.instruction,
                                    hashlib.sha256(self.instruction.encode()).hexdigest(), frames)
            state = build_state(episode, frames, self.tokenizer,
                                [self._features[frame.key].grid_thw for frame in frames],
                                memory_slots=0, include_target=False,
                                merge_size=self.visual.merge_size)
            features = {frame.key: self._features[frame.key] for frame in frames}
            inputs, _, _ = self.reader.prepare([state], features, supervised=False)
            token_ids = greedy_generate(self.backbone, inputs, self.max_new_tokens,
                                        eos_token_ids=self.backbone.generation_config.eos_token_id)
        text = self.tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
        try:
            result = decode_action(text)
        except InvalidActionError as error:
            result = InvalidActionError(str(error))
        self.next_step += 1
        if self.history_mode == "recent":
            self._features = {key: feature for key, feature in self._features.items()
                              if key.step >= step_id - 3}
        self.last_state, self.last_token_ids = state, token_ids.detach()
        self._decisions[step_id] = digest, result
        if isinstance(result, InvalidActionError):
            raise InvalidActionError(str(result))
        return result
