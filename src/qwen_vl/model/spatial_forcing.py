"""Training-only Spatial Forcing for Qwen3.5 VLN.

VGGT is used exclusively to construct an auxiliary representation loss.  Its
features are never inserted into the Qwen forward path, and the stock Qwen3.5
class remains sufficient for inference from a saved checkpoint.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
)

from .geometry_encoders import create_geometry_encoder


VGGT_POSITION_EMBED_SCALE = 0.1
VGGT_POSITION_EMBED_OMEGA = 100.0


def create_aspect_ratio_uv_grid(
    width: int,
    height: int,
    *,
    aspect_ratio: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create the normalized UV grid used by the Spatial-Forcing reference."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("UV-grid dimensions must be positive")
    if aspect_ratio is None:
        aspect_ratio = width / height
    aspect_ratio = float(aspect_ratio)
    if not torch.isfinite(torch.tensor(aspect_ratio)) or aspect_ratio <= 0:
        raise ValueError("UV-grid aspect ratio must be finite and positive")

    diagonal = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diagonal
    span_y = 1.0 / diagonal
    x_coordinates = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        steps=width,
        dtype=dtype,
        device=device,
    )
    y_coordinates = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        steps=height,
        dtype=dtype,
        device=device,
    )
    horizontal, vertical = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    return torch.stack((horizontal, vertical), dim=-1)


def _make_1d_sincos_position_embedding(
    embed_dim: int,
    positions: torch.Tensor,
    *,
    omega_0: float = VGGT_POSITION_EMBED_OMEGA,
) -> torch.Tensor:
    if embed_dim <= 0 or embed_dim % 2:
        raise ValueError(
            "Each 1D sine/cosine embedding dimension must be positive and even"
        )
    frequencies = torch.arange(
        embed_dim // 2,
        dtype=torch.float64,
        device=positions.device,
    )
    frequencies /= embed_dim / 2.0
    frequencies = 1.0 / float(omega_0) ** frequencies
    angles = torch.einsum("m,d->md", positions.reshape(-1), frequencies)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).float()


def position_grid_to_sincos_embedding(
    position_grid: torch.Tensor,
    embed_dim: int,
    *,
    omega_0: float = VGGT_POSITION_EMBED_OMEGA,
) -> torch.Tensor:
    """Convert an ``[H, W, 2]`` UV grid to an ``[H, W, D]`` embedding."""
    if position_grid.ndim != 3 or position_grid.shape[-1] != 2:
        raise ValueError("position_grid must have shape [height, width, 2]")
    if embed_dim <= 0 or embed_dim % 4:
        raise ValueError("2D sine/cosine embedding dimension must be divisible by four")

    height, width, _ = position_grid.shape
    flattened = position_grid.reshape(-1, 2)
    horizontal = _make_1d_sincos_position_embedding(
        embed_dim // 2,
        flattened[:, 0],
        omega_0=omega_0,
    )
    vertical = _make_1d_sincos_position_embedding(
        embed_dim // 2,
        flattened[:, 1],
        omega_0=omega_0,
    )
    return torch.cat((horizontal, vertical), dim=-1).reshape(height, width, embed_dim)


def add_vggt_position_embedding(
    teacher_features: torch.Tensor,
    *,
    teacher_grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
    scale: float = VGGT_POSITION_EMBED_SCALE,
) -> torch.Tensor:
    """Add reference-compatible aspect-ratio-aware UV encoding before pooling."""
    if teacher_features.ndim != 2:
        raise ValueError(
            f"teacher_features must have shape [N, D], got {tuple(teacher_features.shape)}"
        )
    teacher_h, teacher_w = (int(value) for value in teacher_grid_hw)
    image_h, image_w = (int(value) for value in image_hw)
    if min(teacher_h, teacher_w, image_h, image_w) <= 0:
        raise ValueError("Teacher-grid and image dimensions must be positive")
    if teacher_features.shape[0] != teacher_h * teacher_w:
        raise ValueError(
            "VGGT token count does not match its spatial grid before positional encoding: "
            f"tokens={teacher_features.shape[0]}, grid={teacher_h}x{teacher_w}"
        )
    if teacher_features.shape[-1] % 4:
        raise ValueError("VGGT feature dimension must be divisible by four")
    if not torch.isfinite(torch.tensor(float(scale))) or float(scale) < 0:
        raise ValueError(
            "VGGT positional-embedding scale must be finite and nonnegative"
        )

    position_grid = create_aspect_ratio_uv_grid(
        teacher_w,
        teacher_h,
        aspect_ratio=image_w / image_h,
        dtype=teacher_features.dtype,
        device=teacher_features.device,
    )
    position_embedding = position_grid_to_sincos_embedding(
        position_grid,
        teacher_features.shape[-1],
    )
    feature_grid = teacher_features.reshape(
        teacher_h, teacher_w, teacher_features.shape[-1]
    )
    positioned = feature_grid.float() + float(scale) * position_embedding
    if not torch.isfinite(positioned).all():
        raise FloatingPointError("Position-encoded VGGT features contain NaN or Inf")
    return positioned.reshape_as(teacher_features)


class SpatialForcingProjector(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear projection into VGGT space."""

    def __init__(self, student_dim: int, hidden_dim: int, teacher_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(student_dim)
        self.net = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(features))


def resize_teacher_spatial_grid(
    teacher_features: torch.Tensor,
    teacher_grid_hw: tuple[int, int],
    student_grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Resize VGGT patch features as a 2D grid, never as a flat sequence."""
    if teacher_features.ndim != 2:
        raise ValueError(
            f"teacher_features must have shape [N, D], got {tuple(teacher_features.shape)}"
        )
    teacher_h, teacher_w = (int(value) for value in teacher_grid_hw)
    student_h, student_w = (int(value) for value in student_grid_hw)
    if teacher_features.shape[0] != teacher_h * teacher_w:
        raise ValueError(
            "VGGT token count does not match its spatial grid: "
            f"tokens={teacher_features.shape[0]}, grid={teacher_h}x{teacher_w}"
        )
    if min(teacher_h, teacher_w, student_h, student_w) <= 0:
        raise ValueError("Spatial grid dimensions must be positive")

    feature_grid = teacher_features.reshape(
        teacher_h, teacher_w, teacher_features.shape[-1]
    )
    feature_grid = feature_grid.permute(2, 0, 1).unsqueeze(0)
    if (teacher_h, teacher_w) != (student_h, student_w):
        feature_grid = F.interpolate(
            feature_grid.float(),
            size=(student_h, student_w),
            mode="bilinear",
            align_corners=False,
        ).to(teacher_features.dtype)
    return (
        feature_grid.squeeze(0)
        .permute(1, 2, 0)
        .reshape(student_h * student_w, teacher_features.shape[-1])
    )


def spatial_forcing_cosine_loss(
    projected_student: torch.Tensor,
    teacher_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position-wise cosine loss and mean cosine similarity."""
    if projected_student.shape != teacher_features.shape:
        raise ValueError(
            "Student and teacher features must have identical shapes after projection: "
            f"student={tuple(projected_student.shape)}, "
            f"teacher={tuple(teacher_features.shape)}"
        )
    student = F.normalize(projected_student.float(), dim=-1)
    teacher = F.normalize(teacher_features.detach().float(), dim=-1)
    cosine_similarity = (student * teacher).sum(dim=-1)
    return (1.0 - cosine_similarity).mean(), cosine_similarity.mean()


class Qwen3_5ForConditionalGenerationWithSpatialForcing(
    Qwen3_5ForConditionalGeneration
):
    """Qwen3.5 student with a training-only frozen VGGT alignment teacher."""

    accepts_loss_kwargs = False
    _keys_to_ignore_on_load_missing = [
        r"spatial_projector\..*",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.spatial_teacher = None
        self.last_spatial_forcing_metrics = {}
        self._sf_projector_grad_verified = False
        self._sf_student_grad_verified = False

        self.spatial_forcing_enabled = bool(
            getattr(config, "spatial_forcing_enabled", False)
        )
        self.sf_loss_weight = float(getattr(config, "sf_loss_weight", 0.3))
        self.sf_student_layer = int(getattr(config, "sf_student_layer", 24))
        self.sf_teacher_layer = int(getattr(config, "sf_teacher_layer", 23))
        self.sf_use_vggt_pe = bool(getattr(config, "sf_use_vggt_pe", False))
        self.sf_verify_invariants = bool(getattr(config, "sf_verify_invariants", True))
        self.sf_spatial_merge_size = int(
            getattr(config.vision_config, "spatial_merge_size", 2)
        )

        if self.spatial_forcing_enabled:
            student_dim = int(config.text_config.hidden_size)
            teacher_dim = int(getattr(config, "sf_teacher_dim", 2048))
            hidden_dim = int(getattr(config, "sf_projector_hidden_dim", 4096))
            self.spatial_projector = SpatialForcingProjector(
                student_dim=student_dim,
                hidden_dim=hidden_dim,
                teacher_dim=teacher_dim,
            )
            self.spatial_projector.apply(self._init_weights)
            for parameter in self.spatial_projector.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(self._record_projector_gradient)
        else:
            self.spatial_projector = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        teacher_model_path = kwargs.pop("spatial_teacher_model_path", None)
        teacher_cache_dir = kwargs.get("cache_dir")
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        if model.spatial_forcing_enabled:
            if not teacher_model_path:
                raise ValueError(
                    "spatial_teacher_model_path is required for Spatial Forcing training"
                )
            model.initialize_spatial_teacher(
                teacher_model_path, cache_dir=teacher_cache_dir
            )
        return model

    def initialize_spatial_teacher(
        self, teacher_model_path: str, cache_dir: Optional[str] = None
    ) -> None:
        if bool(getattr(self.config, "use_geometry_encoder", False)):
            raise ValueError(
                "Spatial Forcing must not enable SpatialStack geometry fusion"
            )
        if bool(getattr(self.config, "use_geometry_fusion", False)):
            raise ValueError("use_geometry_fusion must be false for Spatial Forcing")

        teacher = create_geometry_encoder(
            encoder_type="vggt",
            model_path=teacher_model_path,
            reference_frame="first",
            freeze_encoder=True,
        )
        teacher.load_model(teacher_model_path, cache_dir=cache_dir)
        teacher.requires_grad_(False)
        teacher.eval()
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise AssertionError("Every VGGT teacher parameter must be frozen")
        self.spatial_teacher = teacher
        self.config.spatial_teacher_model_path = teacher_model_path

    def train(self, mode: bool = True):
        super().train(mode)
        if self.spatial_teacher is not None:
            self.spatial_teacher.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Exclude reproducible frozen VGGT weights from student checkpoints."""
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if key.startswith("spatial_teacher."):
                state.pop(key)
        return state

    def _record_projector_gradient(self, gradient: torch.Tensor):
        if (
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient)
        ):
            self._sf_projector_grad_verified = True
        return gradient

    def _record_student_gradient(self, gradient: torch.Tensor):
        if (
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient)
        ):
            self._sf_student_grad_verified = True
        return gradient

    def _student_capture_layer(self) -> nn.Module:
        # Hugging Face hidden_states[0] is the embedding output, so
        # hidden_states[24] is the output of zero-based decoder layer 23.
        decoder_index = self.sf_student_layer - 1
        layers = self.model.language_model.layers
        if decoder_index < 0 or decoder_index >= len(layers):
            raise ValueError(
                f"sf_student_layer={self.sf_student_layer} is invalid for "
                f"a {len(layers)}-layer Qwen3.5 decoder"
            )
        return layers[decoder_index]

    def _compute_spatial_forcing(
        self,
        student_hidden: torch.Tensor,
        current_image_token_mask: torch.Tensor,
        current_image_grid_thw: torch.Tensor,
        sf_teacher_pixel_values: Sequence[torch.Tensor],
        frame_count: Optional[torch.Tensor],
        current_frame_index: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.spatial_teacher is None or self.spatial_projector is None:
            raise RuntimeError("Spatial Forcing modules were not initialized")
        if current_image_token_mask.dtype != torch.bool:
            current_image_token_mask = current_image_token_mask.bool()
        if student_hidden.shape[:2] != current_image_token_mask.shape:
            raise ValueError(
                "Current-image mask must match Qwen hidden-state sequence shape: "
                f"hidden={tuple(student_hidden.shape)}, "
                f"mask={tuple(current_image_token_mask.shape)}"
            )
        batch_size = student_hidden.shape[0]
        if len(sf_teacher_pixel_values) != batch_size:
            raise ValueError("Each sample must provide exactly one current VGGT frame")
        if current_image_grid_thw.shape != (batch_size, 3):
            raise ValueError(
                "current_image_grid_thw must have shape [batch, 3], got "
                f"{tuple(current_image_grid_thw.shape)}"
            )
        if frame_count is not None and current_frame_index is not None:
            if not torch.equal(current_frame_index, frame_count - 1):
                raise AssertionError("The current observation must be the final frame")

        per_sample_losses = []
        per_sample_cosines = []
        resized_teacher_counts = []
        raw_teacher_counts = []

        for batch_index in range(batch_size):
            grid_t, grid_h, grid_w = (
                int(value) for value in current_image_grid_thw[batch_index].tolist()
            )
            if grid_t != 1:
                raise ValueError(
                    "The initial Spatial Forcing setup expects a single current image"
                )
            student_h = grid_h // self.sf_spatial_merge_size
            student_w = grid_w // self.sf_spatial_merge_size
            expected_student_tokens = student_h * student_w
            student_features = student_hidden[batch_index][
                current_image_token_mask[batch_index]
            ]
            if student_features.shape[0] != expected_student_tokens:
                raise AssertionError(
                    "Current Qwen token count does not match its spatial grid: "
                    f"tokens={student_features.shape[0]}, "
                    f"grid={student_h}x{student_w}"
                )
            if student_features.requires_grad:
                student_features.register_hook(self._record_student_gradient)

            teacher_input = sf_teacher_pixel_values[batch_index]
            if teacher_input.ndim == 3:
                teacher_input = teacher_input.unsqueeze(0)
            if teacher_input.ndim != 4 or teacher_input.shape[0] != 1:
                raise ValueError(
                    "VGGT must receive one current frame per sample, got "
                    f"{tuple(teacher_input.shape)}"
                )
            teacher_input = teacher_input.to(student_features.device)
            teacher_h = int(teacher_input.shape[-2]) // self.spatial_teacher.patch_size
            teacher_w = int(teacher_input.shape[-1]) // self.spatial_teacher.patch_size

            with torch.no_grad():
                teacher_features = self.spatial_teacher.encode_layers(
                    teacher_input,
                    layer_indices=[self.sf_teacher_layer],
                    spatial_merge_size=1,
                    include_camera_token=False,
                )[0][0]
            teacher_features = teacher_features.detach()
            raw_teacher_counts.append(teacher_features.shape[0])
            if self.sf_use_vggt_pe:
                teacher_features = add_vggt_position_embedding(
                    teacher_features,
                    teacher_grid_hw=(teacher_h, teacher_w),
                    image_hw=(
                        int(teacher_input.shape[-2]),
                        int(teacher_input.shape[-1]),
                    ),
                )
            teacher_features = resize_teacher_spatial_grid(
                teacher_features,
                teacher_grid_hw=(teacher_h, teacher_w),
                student_grid_hw=(student_h, student_w),
            )
            resized_teacher_counts.append(teacher_features.shape[0])
            if student_features.shape[0] != teacher_features.shape[0]:
                raise AssertionError(
                    "Student and resized teacher spatial position counts differ"
                )

            projected_student = self.spatial_projector(student_features)
            if projected_student.requires_grad:
                projected_student.register_hook(self._record_projector_gradient)
            sample_loss, sample_cosine = spatial_forcing_cosine_loss(
                projected_student, teacher_features.to(projected_student.device)
            )
            per_sample_losses.append(sample_loss)
            per_sample_cosines.append(sample_cosine)

        sf_loss = torch.stack(per_sample_losses).mean()
        mean_cosine = torch.stack(per_sample_cosines).mean()
        if not torch.isfinite(sf_loss):
            raise FloatingPointError("Spatial Forcing loss contains NaN or Inf")
        metrics = {
            "spatial_forcing_loss": sf_loss.detach(),
            "mean_cosine_similarity": mean_cosine.detach(),
            "current_qwen_token_count": current_image_token_mask.sum(dim=-1)
            .float()
            .mean()
            .detach(),
            "teacher_token_count": torch.tensor(
                resized_teacher_counts,
                device=sf_loss.device,
                dtype=torch.float32,
            ).mean(),
            "teacher_raw_token_count": torch.tensor(
                raw_teacher_counts,
                device=sf_loss.device,
                dtype=torch.float32,
            ).mean(),
            "vggt_pos_embed_enabled": torch.tensor(
                float(self.sf_use_vggt_pe),
                device=sf_loss.device,
                dtype=torch.float32,
            ),
        }
        return sf_loss, metrics

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        current_image_token_mask: Optional[torch.Tensor] = None,
        current_image_grid_thw: Optional[torch.Tensor] = None,
        sf_teacher_pixel_values: Optional[Sequence[torch.Tensor]] = None,
        frame_count: Optional[torch.Tensor] = None,
        current_frame_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Qwen3_5CausalLMOutputWithPast:
        use_spatial_forcing = (
            self.spatial_forcing_enabled and self.training and labels is not None
        )
        captured_hidden = {}
        capture_handle = None
        if use_spatial_forcing:
            if bool(getattr(self.config, "use_geometry_encoder", False)) or bool(
                getattr(self.config, "use_geometry_fusion", False)
            ):
                raise AssertionError(
                    "Geometry fusion cannot run in the Spatial Forcing path"
                )

            def capture_layer_output(_module, _inputs, output):
                captured_hidden["value"] = (
                    output[0] if isinstance(output, tuple) else output
                )

            capture_handle = self._student_capture_layer().register_forward_hook(
                capture_layer_output
            )

        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        finally:
            if capture_handle is not None:
                capture_handle.remove()

        if not use_spatial_forcing:
            self.last_spatial_forcing_metrics = {}
            return outputs
        if "value" not in captured_hidden:
            raise RuntimeError("Failed to capture the requested Qwen3.5 hidden state")
        if any(
            value is None
            for value in (
                current_image_token_mask,
                current_image_grid_thw,
                sf_teacher_pixel_values,
            )
        ):
            raise ValueError("Spatial Forcing batch metadata is incomplete")

        navigation_loss = outputs.loss
        if navigation_loss is None or not torch.isfinite(navigation_loss):
            raise FloatingPointError("Navigation loss is missing, NaN, or Inf")
        sf_loss, metrics = self._compute_spatial_forcing(
            student_hidden=captured_hidden["value"],
            current_image_token_mask=current_image_token_mask,
            current_image_grid_thw=current_image_grid_thw,
            sf_teacher_pixel_values=sf_teacher_pixel_values,
            frame_count=frame_count,
            current_frame_index=current_frame_index,
        )
        total_loss = navigation_loss + self.sf_loss_weight * sf_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Total training loss contains NaN or Inf")

        metrics.update(
            navigation_loss=navigation_loss.detach(),
            total_loss=total_loss.detach(),
        )
        self.last_spatial_forcing_metrics = metrics
        return Qwen3_5CausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
