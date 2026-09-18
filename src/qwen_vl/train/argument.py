from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-4B")
    attn_implementation: str = field(default="flash_attention_2")
    tune_mm_llm: bool = field(default=True)
    tune_mm_mlp: bool = field(default=True)
    tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_config: str = field(
        default="configs/datasets/newton_r2r_v0.json"
    )
    max_history_frames: int = field(default=8)
    max_pixels: int = field(default=576 * 28 * 28)
    min_pixels: int = field(default=16 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)
    sparse_action_logits: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=12_800)
    mm_projector_lr: Optional[float] = field(default=1e-5)
    group_by_modality_length: bool = field(default=True)
    save_final_model: bool = field(default=True)
