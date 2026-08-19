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

from .depth_supervision import (
    GeoVRDepthHead,
    compute_geovr_depth_loss,
    finite_depth_statistics,
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
        r"student_depth_head\..*",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.spatial_teacher = None
        self.last_spatial_forcing_metrics = {}
        self._sf_projector_grad_verified = False
        self._sf_student_grad_verified = False
        self._depth_head_grad_verified = False
        self._depth_student_grad_verified = False

        self.spatial_forcing_enabled = bool(
            getattr(config, "spatial_forcing_enabled", False)
        )
        self.sf_loss_weight = float(getattr(config, "sf_loss_weight", 0.3))
        self.sf_student_layer = int(getattr(config, "sf_student_layer", 24))
        self.sf_teacher_layer = int(getattr(config, "sf_teacher_layer", 23))
        self.sf_use_vggt_pe = bool(getattr(config, "sf_use_vggt_pe", False))
        self.sf_verify_invariants = bool(getattr(config, "sf_verify_invariants", True))
        self.sf_multiframe_teacher = bool(
            getattr(config, "sf_multiframe_teacher", False)
        )
        self.sf_spatial_merge_size = int(
            getattr(config.vision_config, "spatial_merge_size", 2)
        )
        self.depth_supervision_enabled = bool(
            getattr(config, "depth_supervision_enabled", False)
        )
        self.depth_loss_weight = float(getattr(config, "depth_loss_weight", 0.05))
        self.depth_student_layers = tuple(
            int(layer)
            for layer in getattr(config, "depth_student_layers", [7, 16, 24, 32])
        )
        self.depth_gradient_scales = tuple(
            int(scale)
            for scale in getattr(config, "depth_gradient_scales", [1, 2, 4, 8])
        )
        self.depth_outlier_keep_ratio = float(
            getattr(config, "depth_outlier_keep_ratio", 0.98)
        )
        self.depth_use_teacher_confidence = bool(
            getattr(config, "depth_use_teacher_confidence", False)
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

        if self.depth_supervision_enabled:
            if not self.spatial_forcing_enabled:
                raise ValueError("Depth supervision requires Spatial Forcing v0")
            if self.sf_multiframe_teacher:
                raise ValueError("Depth supervision requires a current-frame-only teacher")
            if self.depth_student_layers != (7, 16, 24, 32):
                raise ValueError(
                    "Depth supervision requires layers (7, 16, 24, 32), got "
                    f"{self.depth_student_layers}"
                )
            if self.depth_loss_weight < 0:
                raise ValueError("depth_loss_weight must be nonnegative")
            if self.depth_use_teacher_confidence:
                raise ValueError(
                    "Teacher confidence weighting is excluded from the v4 ablation"
                )
            vision_patch_size = int(getattr(config.vision_config, "patch_size", 14))
            self.student_depth_head = GeoVRDepthHead(
                dim_in=int(config.text_config.hidden_size),
                patch_size=vision_patch_size * self.sf_spatial_merge_size,
                target_patch_size=14 * 2,
            )
            # GeoVR relies on the default PyTorch initialization of its newly
            # attached DenseHead. Do not overwrite it with Qwen initialization.
            for parameter in self.student_depth_head.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(self._record_depth_head_gradient)
        else:
            self.student_depth_head = None

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
            enable_depth=self.depth_supervision_enabled,
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Restore a student checkpoint while reusing the initialized teacher.

        DeepSpeed restores module state strictly. The frozen VGGT is
        deliberately omitted by :meth:`state_dict`, so its missing keys are
        the only strict-load exception; all student/trainable missing or
        unexpected keys still fail immediately.
        """
        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        missing_keys = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("spatial_teacher.")
        ]
        unexpected_keys = list(incompatible.unexpected_keys)
        if strict and (missing_keys or unexpected_keys):
            messages = []
            if unexpected_keys:
                messages.append(
                    "Unexpected key(s) in state_dict: {}.".format(
                        ", ".join(f'"{key}"' for key in unexpected_keys)
                    )
                )
            if missing_keys:
                messages.append(
                    "Missing key(s) in state_dict: {}.".format(
                        ", ".join(f'"{key}"' for key in missing_keys)
                    )
                )
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(messages)
                )
            )
        return type(incompatible)(missing_keys, unexpected_keys)

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

    def _record_depth_head_gradient(self, gradient: torch.Tensor):
        if (
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient)
        ):
            self._depth_head_grad_verified = True
        return gradient

    def _record_depth_student_gradient(self, gradient: torch.Tensor):
        if (
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient)
        ):
            self._depth_student_grad_verified = True
        return gradient

    def _student_capture_module(self, hidden_state_index: int) -> nn.Module:
        """Map a Hugging Face hidden-state index to its producing module.

        Entry zero is the input embedding. Entries 1..N-1 are raw decoder
        block outputs, while entry N is the final block output after the
        language model's final RMSNorm.
        """
        layers = self.model.language_model.layers
        hidden_state_index = int(hidden_state_index)
        if hidden_state_index == len(layers):
            return self.model.language_model.norm
        decoder_index = hidden_state_index - 1
        if decoder_index < 0 or decoder_index >= len(layers):
            raise ValueError(
                f"hidden_state_index={hidden_state_index} is invalid for "
                f"a {len(layers)}-layer Qwen3.5 decoder"
            )
        return layers[decoder_index]

    def _student_capture_layer(self) -> nn.Module:
        # Existing v0 convention: hidden_states[24] is decoder block 23 output.
        return self._student_capture_module(self.sf_student_layer)

    def _compute_spatial_forcing(
        self,
        student_hidden: torch.Tensor,
        current_image_token_mask: torch.Tensor,
        current_image_grid_thw: torch.Tensor,
        sf_teacher_pixel_values: Sequence[torch.Tensor],
        frame_count: Optional[torch.Tensor],
        current_frame_index: Optional[torch.Tensor],
        depth_student_hidden: Optional[dict[int, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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

        use_depth = self.depth_supervision_enabled
        if use_depth:
            if self.student_depth_head is None:
                raise RuntimeError("Student depth head was not initialized")
            if depth_student_hidden is None:
                raise RuntimeError("Depth hidden states were not captured")
            missing_layers = set(self.depth_student_layers) - set(depth_student_hidden)
            if missing_layers:
                raise RuntimeError(
                    f"Missing captured Qwen depth layers: {sorted(missing_layers)}"
                )
            if depth_student_hidden[self.sf_student_layer] is not student_hidden:
                raise AssertionError(
                    "Layer 24 must be captured once and reused by SF and depth"
                )

        per_sample_losses = []
        per_sample_cosines = []
        resized_teacher_counts = []
        raw_teacher_counts = []
        depth_losses = []
        depth_regression_losses = []
        depth_gradient_losses = []
        depth_maes = []
        depth_valid_fractions = []
        depth_target_heights = []
        depth_target_widths = []
        teacher_pad_heights = []
        teacher_pad_widths = []
        teacher_invalid_fractions = {
            "nan": [],
            "inf": [],
            "nonpositive": [],
        }
        teacher_depth_statistics = {}
        predicted_depth_statistics = {}
        teacher_confidence_means = []

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
                if use_depth:
                    teacher_outputs = (
                        self.spatial_teacher.encode_features_and_depth(
                            teacher_input,
                            layer_indices=[self.sf_teacher_layer],
                            spatial_merge_size=1,
                            include_camera_token=False,
                        )
                    )
                    teacher_features = teacher_outputs["sf_features"][0][0]
                    teacher_h, teacher_w = teacher_outputs["teacher_grid_hw"]
                    teacher_image_hw = teacher_outputs["teacher_image_hw"]
                    teacher_pad_heights.append(
                        float(teacher_outputs["teacher_padding_hw"][0])
                    )
                    teacher_pad_widths.append(
                        float(teacher_outputs["teacher_padding_hw"][1])
                    )
                else:
                    teacher_outputs = None
                    teacher_image_hw = (
                        int(teacher_input.shape[-2]),
                        int(teacher_input.shape[-1]),
                    )
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
                    image_hw=teacher_image_hw,
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

            if use_depth:
                teacher_depth = teacher_outputs["depth"][0]
                teacher_confidence = teacher_outputs["depth_conf"][0]
                if teacher_depth.ndim != 2:
                    raise AssertionError(
                        "Current-frame VGGT pseudo-depth must be two-dimensional, "
                        f"got {tuple(teacher_depth.shape)}"
                    )
                if teacher_confidence.shape != teacher_depth.shape:
                    raise AssertionError(
                        "VGGT depth confidence must match pseudo-depth shape"
                    )

                selected_features = []
                for layer_index in self.depth_student_layers:
                    layer_hidden = depth_student_hidden[layer_index]
                    if layer_hidden.shape[:2] != current_image_token_mask.shape:
                        raise ValueError(
                            f"Layer {layer_index} shape does not match the current-token mask"
                        )
                    current_features = layer_hidden[batch_index][
                        current_image_token_mask[batch_index]
                    ]
                    if current_features.shape[0] != expected_student_tokens:
                        raise AssertionError(
                            "Depth-layer current-token count does not match the grid: "
                            f"layer={layer_index}, tokens={current_features.shape[0]}, "
                            f"expected={expected_student_tokens}"
                        )
                    if current_features.requires_grad:
                        current_features.register_hook(
                            self._record_depth_student_gradient
                        )
                    selected_features.append(current_features.unsqueeze(0))

                predicted_depth = self.student_depth_head(
                    selected_features,
                    student_grid_hw=(student_h, student_w),
                    image_hw=(
                        int(teacher_input.shape[-2]),
                        int(teacher_input.shape[-1]),
                    ),
                    target_hw=tuple(int(value) for value in teacher_depth.shape),
                )[0]
                if predicted_depth.requires_grad:
                    # Register on the live output rather than relying only on
                    # constructor-time parameter hooks. DeepSpeed ZeRO may
                    # replace/partition parameter objects during wrapping,
                    # while this tensor is created inside the wrapped forward.
                    predicted_depth.register_hook(
                        self._record_depth_head_gradient
                    )
                depth_output = compute_geovr_depth_loss(
                    predicted_depth.float(),
                    teacher_depth.float(),
                    gradient_scales=self.depth_gradient_scales,
                    outlier_keep_ratio=self.depth_outlier_keep_ratio,
                )
                depth_losses.append(depth_output.loss)
                depth_regression_losses.append(depth_output.regression)
                depth_gradient_losses.append(depth_output.gradient)
                depth_maes.append(depth_output.mae)
                depth_valid_fractions.append(depth_output.valid_fraction)
                depth_target_heights.append(float(teacher_depth.shape[-2]))
                depth_target_widths.append(float(teacher_depth.shape[-1]))

                teacher_invalid_fractions["nan"].append(
                    torch.isnan(teacher_depth).float().mean()
                )
                teacher_invalid_fractions["inf"].append(
                    torch.isinf(teacher_depth).float().mean()
                )
                teacher_invalid_fractions["nonpositive"].append(
                    (torch.isfinite(teacher_depth) & (teacher_depth <= 0))
                    .float()
                    .mean()
                )
                for name, value in finite_depth_statistics(teacher_depth).items():
                    teacher_depth_statistics.setdefault(name, []).append(value)
                for name, value in finite_depth_statistics(predicted_depth).items():
                    predicted_depth_statistics.setdefault(name, []).append(value)
                finite_confidence = teacher_confidence[
                    torch.isfinite(teacher_confidence)
                ]
                teacher_confidence_means.append(
                    finite_confidence.float().mean()
                    if finite_confidence.numel()
                    else torch.tensor(
                        float("nan"), device=teacher_confidence.device
                    )
                )

        sf_loss = torch.stack(per_sample_losses).mean()
        mean_cosine = torch.stack(per_sample_cosines).mean()
        if not torch.isfinite(sf_loss):
            raise FloatingPointError("Spatial Forcing loss contains NaN or Inf")
        metrics = {
            "spatial_forcing_loss": sf_loss.detach(),
            "sf_weighted_loss": (self.sf_loss_weight * sf_loss).detach(),
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
        if use_depth:
            depth_loss = torch.stack(depth_losses).mean()
            depth_regression_loss = torch.stack(depth_regression_losses).mean()
            depth_gradient_loss = torch.stack(depth_gradient_losses).mean()
            if not torch.isfinite(depth_loss):
                raise FloatingPointError("Depth loss contains NaN or Inf")

            def mean_values(values):
                return torch.stack(
                    [value.detach().float().to(sf_loss.device) for value in values]
                ).mean()

            metrics.update(
                depth_loss=depth_loss.detach(),
                depth_reg_loss=depth_regression_loss.detach(),
                depth_grad_loss=depth_gradient_loss.detach(),
                depth_weighted_loss=(self.depth_loss_weight * depth_loss).detach(),
                depth_mae=mean_values(depth_maes),
                depth_valid_fraction=mean_values(depth_valid_fractions),
                depth_target_height=torch.tensor(
                    depth_target_heights,
                    device=sf_loss.device,
                    dtype=torch.float32,
                ).mean(),
                depth_target_width=torch.tensor(
                    depth_target_widths,
                    device=sf_loss.device,
                    dtype=torch.float32,
                ).mean(),
                teacher_depth_nan_fraction=mean_values(
                    teacher_invalid_fractions["nan"]
                ),
                teacher_depth_inf_fraction=mean_values(
                    teacher_invalid_fractions["inf"]
                ),
                teacher_depth_nonpositive_fraction=mean_values(
                    teacher_invalid_fractions["nonpositive"]
                ),
                teacher_depth_confidence_mean=mean_values(
                    teacher_confidence_means
                ),
                vggt_aggregator_calls_per_sample=torch.tensor(
                    1.0, device=sf_loss.device
                ),
                vggt_teacher_pad_height=torch.tensor(
                    teacher_pad_heights,
                    device=sf_loss.device,
                    dtype=torch.float32,
                ).mean(),
                vggt_teacher_pad_width=torch.tensor(
                    teacher_pad_widths,
                    device=sf_loss.device,
                    dtype=torch.float32,
                ).mean(),
                depth_layer_24_reused=torch.tensor(1.0, device=sf_loss.device),
            )
            for name, values in teacher_depth_statistics.items():
                metrics[f"teacher_depth_{name}"] = mean_values(values)
            for name, values in predicted_depth_statistics.items():
                metrics[f"pred_depth_{name}"] = mean_values(values)
        else:
            depth_loss = sf_loss.new_zeros(())
        return sf_loss, depth_loss, metrics

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
        captured_hidden: dict[int, torch.Tensor] = {}
        capture_handles = []
        if use_spatial_forcing:
            if bool(getattr(self.config, "use_geometry_encoder", False)) or bool(
                getattr(self.config, "use_geometry_fusion", False)
            ):
                raise AssertionError(
                    "Geometry fusion cannot run in the Spatial Forcing path"
                )

            capture_indices = {self.sf_student_layer}
            if self.depth_supervision_enabled:
                capture_indices.update(self.depth_student_layers)

            for hidden_state_index in sorted(capture_indices):
                def capture_layer_output(
                    _module,
                    _inputs,
                    output,
                    *,
                    index=hidden_state_index,
                ):
                    captured_hidden[index] = (
                        output[0] if isinstance(output, tuple) else output
                    )

                capture_handles.append(
                    self._student_capture_module(
                        hidden_state_index
                    ).register_forward_hook(capture_layer_output)
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
            for capture_handle in capture_handles:
                capture_handle.remove()

        if not use_spatial_forcing:
            self.last_spatial_forcing_metrics = {}
            return outputs
        if self.sf_student_layer not in captured_hidden:
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
        sf_loss, depth_loss, metrics = self._compute_spatial_forcing(
            student_hidden=captured_hidden[self.sf_student_layer],
            current_image_token_mask=current_image_token_mask,
            current_image_grid_thw=current_image_grid_thw,
            sf_teacher_pixel_values=sf_teacher_pixel_values,
            frame_count=frame_count,
            current_frame_index=current_frame_index,
            depth_student_hidden=(
                captured_hidden if self.depth_supervision_enabled else None
            ),
        )
        total_loss = (
            navigation_loss
            + self.sf_loss_weight * sf_loss
            + self.depth_loss_weight * depth_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Total training loss contains NaN or Inf")

        metrics.update(
            navigation_loss=navigation_loss.detach(),
            total_loss=total_loss.detach(),
            configured_sf_loss_weight=torch.tensor(
                self.sf_loss_weight, device=total_loss.device
            ),
            configured_depth_loss_weight=torch.tensor(
                self.depth_loss_weight, device=total_loss.device
            ),
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
