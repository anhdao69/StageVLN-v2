"""Explicit observation, prompt and feature identities for recurrent navigation."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class FrameKey:
    dataset: str
    episode: str
    step: int

@dataclass(frozen=True)
class FrameRecord:
    key: FrameKey
    image_path: str
    action: str | None
    source_record_id: str

@dataclass(frozen=True)
class EpisodeRecord:
    episode: str
    instruction: str
    instruction_sha256: str
    frames: tuple[FrameRecord, ...]

@dataclass(frozen=True)
class ImageSpan:
    key: FrameKey
    start: int
    stop: int
    grid_thw: tuple[int, int, int]
    is_current: bool

@dataclass
class TokenizedState:
    input_ids: torch.Tensor
    labels: torch.Tensor
    image_spans: tuple[ImageSpan, ...]
    memory_span: tuple[int, int] | None
    action_start: int
    frame_keys: tuple[FrameKey, ...]

@dataclass
class FrameFeatures:
    key: FrameKey
    premerge: torch.Tensor
    grid_thw: tuple[int, int, int]
    merged: torch.Tensor | None
    original_hw: tuple[int, int] = (0, 0)
    encoder_fingerprint: str = ''

@dataclass
class WriterStep:
    state: torch.Tensor
    levels: tuple[torch.Tensor, ...]
