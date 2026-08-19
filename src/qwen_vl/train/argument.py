import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-4B")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # ``geometry_encoder_path`` keeps the launcher's established CLI name, but
    # the model is a loss-only teacher and never enters Qwen's inference path.
    geometry_encoder_path: str = field(default="facebook/VGGT-1B")
    use_geometry_encoder: bool = field(default=False)
    use_geometry_fusion: bool = field(default=False)
    spatial_forcing_enabled: bool = field(default=False)
    sf_loss_weight: float = field(default=0.3)
    sf_student_layer: int = field(default=24)
    sf_teacher_layer: int = field(default=23)
    sf_use_vggt_pe: bool = field(default=False)
    sf_projector_hidden_dim: int = field(default=4096)
    sf_verify_invariants: bool = field(default=True)
    sf_multiframe_teacher: bool = field(default=False)

    depth_supervision_enabled: bool = field(default=False)
    depth_loss_weight: float = field(default=0.05)
    depth_student_layers: list[int] = field(
        default_factory=lambda: [7, 16, 24, 32]
    )
    depth_loss_type: str = field(default="geo_depth")
    depth_gradient_scales: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8]
    )
    depth_outlier_keep_ratio: float = field(default=0.98)
    depth_use_teacher_confidence: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    dataset_config: Optional[str] = field(default=None)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)
    janusvln_data_root: str = field(default="data/JanusVLN_data")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    depth_head_lr: Optional[float] = field(default=1e-5)
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
