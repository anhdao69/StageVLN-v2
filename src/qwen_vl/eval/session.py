"""A fixed-policy actor: one observation write, then a fresh action generation.

Create a FrameLoader using the exported processor, construct PolicySession, call
reset(episode_id, instruction), then observe(step_id, rgb) in chronological order.
RGB may be a PIL image, uint8 HWC NumPy array, loader-root-relative image path,
or FrameRecord. FrameRecord action/source labels never enter inference.

Recent visual features are safe to retain only while policy weights are fixed.
Sessions may share a policy sequentially; concurrent calls are not supported.
"""
from contextlib import nullcontext
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from qwen_vl.contracts import EpisodeRecord, FrameKey, FrameRecord
from qwen_vl.data.prompting import build_state


ACTIONS = ('MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP')


class InvalidActionError(ValueError):
    """The model generated text outside the four-action protocol."""


def decode_action(text: str) -> str:
    """Accept one exact primitive action with optional surrounding whitespace."""
    action = text.strip()
    if action not in ACTIONS:
        raise InvalidActionError(f'Invalid generated action: {text!r}')
    return action


@torch.inference_mode()
def greedy_generate(backbone, inputs, max_new_tokens=16, eos_token_ids=None):
    """Generate batch-one token IDs from a prepared reader prefix.

    Returns only generated IDs, shape [1, n], including a generated EOS. The
    prefix uses inputs_embeds and explicit 3D RoPE positions from reader.prepare.
    Qwen's native 4-axis convention adds sequential text positions for masking.
    For token j fed after prefill, cache_position=L+j and every RoPE coordinate
    is max(prefix_positions)+1+j. No image or writer call occurs in this loop.
    The cache is a local variable and is discarded at every decision boundary.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    embeddings = inputs['inputs_embeds']
    positions = inputs['position_ids']
    mask = inputs['attention_mask']
    if embeddings.ndim != 3 or embeddings.shape[0] != 1 or not bool(mask.all()):
        raise ValueError('Generation requires one unpadded prefix')
    length = embeddings.shape[1]
    if positions.shape != (3, 1, length):
        raise ValueError('Reader must supply explicit three-axis prefix positions')
    if inputs.get('past_key_values') is not None:
        raise ValueError('Navigation decisions cannot reuse a generation cache')
    if eos_token_ids is None:
        eos_token_ids = backbone.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = ()
    if isinstance(eos_token_ids, int):
        eos_token_ids = (eos_token_ids,)
    eos = set(eos_token_ids)
    device = embeddings.device
    next_position = int(positions.max()) + 1
    text_positions = torch.arange(length, device=device).view(1, 1, -1)
    prefill = dict(inputs)
    prefill.update(position_ids=torch.cat((text_positions, positions), dim=0),
                   cache_position=torch.arange(length, device=device),
                   use_cache=True, past_key_values=None, logits_to_keep=1,
                   return_dict=True)
    backbone.model.rope_deltas = None
    generated = []
    try:
        output = backbone(**prefill)
        for index in range(max_new_tokens):
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            if int(token.item()) in eos or index + 1 == max_new_tokens:
                break
            cache_position = torch.tensor([length + index], device=device)
            rope = torch.full((3, 1, 1), next_position + index, device=device, dtype=torch.long)
            positions = torch.cat((cache_position.view(1, 1, 1), rope), dim=0)
            output = backbone(input_ids=token, attention_mask=torch.ones(
                (1, length + index + 1), device=device, dtype=mask.dtype),
                position_ids=positions, cache_position=cache_position,
                past_key_values=output.past_key_values, use_cache=True,
                logits_to_keep=1, return_dict=True)
        return torch.cat(generated, dim=1)
    finally:
        # Explicit positions make this irrelevant, but leave no stale native
        # generation metadata for another actor sharing the same backbone.
        backbone.model.rope_deltas = None


class PolicySession:
    """Caller-owned recurrent memory and recent history for one active episode.

    Duplicate observations, including old steps, return their original action.
    Conflicting RGB, changed instruction for the same episode, and skipped new
    steps fail. A decode failure is remembered so its observation is not written
    twice. Call reset with a new episode identity to start another instruction.
    last_state/last_token_ids expose the most recent prefix/output for auditing.
    """

    def __init__(self, policy, loader, recent=0, max_new_tokens=16, bf16=True):
        if recent not in (0, 4):
            raise ValueError('Only recent=0 or recent=4 is supported')
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('max_new_tokens must be a positive integer')
        self.policy = policy.eval()
        self.loader = loader
        self.recent, self.max_new_tokens, self.bf16 = recent, max_new_tokens, bf16
        self.episode_id = None
        self.instruction = None
        self.memory = None
        self.instruction_features = None
        self.next_step = 0
        self.last_state = self.last_token_ids = None
        self._frames = []
        self._features = {}
        self._decisions = {}

    @property
    def recent_keys(self):
        return tuple(frame.key for frame in self._frames)

    def _autocast(self):
        if self.bf16 and self.policy.device.type == 'cuda':
            return torch.autocast('cuda', dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def reset(self, episode_id, instruction):
        episode_id = str(episode_id)
        if not episode_id or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Episode identity and instruction must be nonempty')
        if self.episode_id == episode_id and self.instruction != instruction:
            raise ValueError('Conflicting instruction for the same episode')
        with self._autocast():
            instruction_features = self.policy.encode_instruction(instruction)
        self.episode_id, self.instruction = episode_id, instruction
        self.instruction_features = instruction_features
        self.memory = None
        self.next_step = 0
        self._frames, self._features, self._decisions = [], {}, {}
        self.last_state = self.last_token_ids = None

    def _observation(self, step_id, rgb):
        frame = FrameRecord(FrameKey('r2r', self.episode_id, step_id), '', None, '')
        from_file = isinstance(rgb, (FrameRecord, str, Path))
        if isinstance(rgb, FrameRecord):
            if rgb.key.episode != self.episode_id or rgb.key.step != step_id:
                raise ValueError('Observation identity conflicts with the active episode/step')
            frame = replace(rgb, action=None)
        elif isinstance(rgb, (str, Path)):
            frame = replace(frame, image_path=str(rgb))
        if from_file:
            with Image.open(self.loader.root / frame.image_path) as source:
                image = source.convert('RGB')
        elif isinstance(rgb, Image.Image):
            image = rgb.convert('RGB')
        elif isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3:
            image = Image.fromarray(rgb)
        else:
            raise TypeError('RGB must be a PIL image, uint8 HWC array, path, or FrameRecord')
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        return frame, image, digest, from_file

    @torch.inference_mode()
    def generate_greedy(self, state, features, memory=None):
        """Generate raw IDs from one already-written state, without another write."""
        inputs, _, _ = self.policy.reader.prepare([state], features, [memory], supervised=False)
        return greedy_generate(self.policy.backbone, inputs, self.max_new_tokens,
                               eos_token_ids=self.policy.tokenizer.eos_token_id)

    @torch.inference_mode()
    def observe(self, step_id, rgb):
        if self.episode_id is None:
            raise ValueError('reset must establish an episode before observing')
        if type(step_id) is not int or step_id < 0:
            raise ValueError('Observation step must be a nonnegative integer')
        frame, image, digest, from_file = self._observation(step_id, rgb)
        if step_id in self._decisions:
            previous_digest, result = self._decisions[step_id]
            if previous_digest != digest:
                raise ValueError('Observation RGB conflicts with the previously observed step')
            if isinstance(result, InvalidActionError):
                raise InvalidActionError(str(result))
            return result
        if step_id != self.next_step:
            raise ValueError('New observations must be chronological without skipped steps')
        if from_file:
            # A FrameKey may recur after reset; do not reuse preprocessing of an
            # earlier image at that key if the caller has replaced its contents.
            self.loader.cache.pop(frame.key, None)
            pixels, grid = self.loader.get(frame)
        else:
            processed = self.loader.processor.preprocess([image], return_tensors='pt')
            pixels, grid = processed['pixel_values'], processed['image_grid_thw'][0]
        with self._autocast():
            feature = self.policy.visual.encode(pixels.to(self.policy.device),
                grid.unsqueeze(0).to(self.policy.device), [frame.key])[0]
            features = dict(self._features)
            features[frame.key] = feature
            memory = self.policy.write(self.memory, feature, self.instruction_features)
            frames = self._frames + [frame]
            episode = EpisodeRecord(self.episode_id, self.instruction,
                                    hashlib.sha256(self.instruction.encode()).hexdigest(), tuple(frames))
            state = build_state(episode, frames, self.policy.tokenizer,
                                [features[f.key].grid_thw for f in frames],
                                self.policy.memory_slots, include_target=False,
                                merge_size=self.policy.visual.merge_size)
            adapted = self.policy.memory_adapter(memory)[0] if self.policy.memory_enabled else None
            token_ids = self.generate_greedy(state, features, adapted)
        text = self.policy.tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
        try:
            result = decode_action(text)
        except ValueError as error:
            # Do not retain a traceback: it holds the whole observation frame
            # and would keep discarded visual features alive across the episode.
            result = InvalidActionError(str(error))
        self.memory = memory.detach().float() if memory is not None else None
        self._frames = frames[-self.recent:] if self.recent else []
        self._features = {f.key: features[f.key] for f in self._frames}
        self.next_step += 1
        self.last_state, self.last_token_ids = state, token_ids.detach()
        self._decisions[step_id] = digest, result
        if isinstance(result, InvalidActionError):
            raise InvalidActionError(str(result))
        return result
