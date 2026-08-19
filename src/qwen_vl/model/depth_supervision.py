"""GeoVR-style training-only dense depth supervision.

The dense decoder and robust loss in this module are adapted from the supplied
GeoVR implementation.  Unlike GeoVR's original head, ``GeoVRDepthHead`` accepts
exactly four already-selected Qwen hidden states in shallow-to-deep order; it
never indexes a full hidden-state tuple by absolute decoder layer number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_sincos_position_embedding(
    embed_dim: int,
    positions: torch.Tensor,
    omega_0: float = 100.0,
) -> torch.Tensor:
    """Return GeoVR's one-dimensional sine/cosine position embedding."""
    if embed_dim <= 0 or embed_dim % 2:
        raise ValueError("The 1D position embedding dimension must be positive and even")
    omega_dtype = torch.float32 if positions.device.type == "mps" else torch.float64
    omega = torch.arange(
        embed_dim // 2,
        dtype=omega_dtype,
        device=positions.device,
    )
    omega /= embed_dim / 2.0
    omega = 1.0 / float(omega_0) ** omega
    angles = torch.einsum("m,d->md", positions.reshape(-1), omega)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).float()


def position_grid_to_embedding(
    position_grid: torch.Tensor,
    embed_dim: int,
    omega_0: float = 100.0,
) -> torch.Tensor:
    """Convert an ``[H, W, 2]`` normalized UV grid to ``[H, W, C]``."""
    if position_grid.ndim != 3 or position_grid.shape[-1] != 2:
        raise ValueError("position_grid must have shape [height, width, 2]")
    if embed_dim <= 0 or embed_dim % 4:
        raise ValueError("The 2D position embedding dimension must be divisible by four")
    height, width, _ = position_grid.shape
    flattened = position_grid.reshape(-1, 2)
    horizontal = make_sincos_position_embedding(
        embed_dim // 2, flattened[:, 0], omega_0=omega_0
    )
    vertical = make_sincos_position_embedding(
        embed_dim // 2, flattened[:, 1], omega_0=omega_0
    )
    return torch.cat((horizontal, vertical), dim=-1).reshape(
        height, width, embed_dim
    )


def create_normalized_uv_grid(
    width: int,
    height: int,
    *,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create the deterministic, diagonal-normalized UV grid used by GeoVR."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("UV-grid dimensions must be positive")
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    aspect_ratio = float(aspect_ratio)
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("UV-grid aspect ratio must be finite and positive")

    diagonal = math.sqrt(aspect_ratio**2 + 1.0)
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
    horizontal, vertical = torch.meshgrid(
        x_coordinates, y_coordinates, indexing="xy"
    )
    return torch.stack((horizontal, vertical), dim=-1)


def custom_interpolate(
    tensor: torch.Tensor,
    *,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Interpolate in bounded chunks to avoid PyTorch's INT_MAX limit."""
    if size is None:
        if scale_factor is None:
            raise ValueError("Either size or scale_factor is required")
        size = (
            int(tensor.shape[-2] * scale_factor),
            int(tensor.shape[-1] * scale_factor),
        )
    size = (int(size[0]), int(size[1]))
    if min(size) <= 0:
        raise ValueError(f"Interpolation size must be positive, got {size}")
    if tuple(tensor.shape[-2:]) == size:
        return tensor

    int_max = 1_610_612_736
    output_elements = size[0] * size[1] * tensor.shape[0] * tensor.shape[1]
    if output_elements <= int_max:
        return F.interpolate(
            tensor,
            size=size,
            mode=mode,
            align_corners=align_corners,
        )
    chunks = torch.chunk(tensor, chunks=(output_elements // int_max) + 1, dim=0)
    return torch.cat(
        [
            F.interpolate(
                chunk,
                size=size,
                mode=mode,
                align_corners=align_corners,
            )
            for chunk in chunks
        ],
        dim=0,
    ).contiguous()


def _make_resize_layer(channels: int, resize_scale: float) -> nn.Module:
    if resize_scale == 1.0:
        return nn.Identity()
    if resize_scale == 0.5:
        return nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    upsample_scale = int(resize_scale)
    return nn.ConvTranspose2d(
        channels,
        channels,
        kernel_size=upsample_scale,
        stride=upsample_scale,
        padding=0,
    )


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=1)
        self.activation = activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.activation(tensor)
        output = self.conv1(output)
        output = self.activation(output)
        output = self.conv2(output)
        return output + tensor


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, *, has_residual: bool = True) -> None:
        super().__init__()
        activation = nn.ReLU(inplace=False)
        self.has_residual = has_residual
        if has_residual:
            self.residual_unit = ResidualConvUnit(features, activation)
        self.output_unit = ResidualConvUnit(features, activation)
        self.output_projection = nn.Conv2d(features, features, kernel_size=1)

    def forward(
        self,
        tensor: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        size: tuple[int, int],
    ) -> torch.Tensor:
        output = tensor
        if self.has_residual:
            if residual is None:
                raise ValueError("A residual feature map is required")
            output = output + self.residual_unit(residual)
        output = self.output_unit(output)
        output = custom_interpolate(
            output,
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        return self.output_projection(output)


class GeoVRDepthHead(nn.Module):
    """GeoVR DenseHead adapted to a four-element selected-feature list."""

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 28,
        target_patch_size: int = 28,
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
    ) -> None:
        super().__init__()
        if patch_size <= 0 or patch_size % 4:
            raise ValueError("patch_size must be positive and divisible by four")
        if target_patch_size <= 0 or target_patch_size % 4:
            raise ValueError(
                "target_patch_size must be positive and divisible by four"
            )
        if len(out_channels) != 4:
            raise ValueError("GeoVRDepthHead requires exactly four channel levels")
        if any(int(channel) <= 0 for channel in out_channels):
            raise ValueError("All projected channel dimensions must be positive")

        self.patch_size = int(patch_size)
        self.target_patch_size = int(target_patch_size)
        self.final_shuffle_factor = self.target_patch_size // 4
        self.norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(dim_in, int(channel), kernel_size=1)
                for channel in out_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                _make_resize_layer(int(out_channels[0]), 4.0),
                _make_resize_layer(int(out_channels[1]), 2.0),
                _make_resize_layer(int(out_channels[2]), 1.0),
                _make_resize_layer(int(out_channels[3]), 0.5),
            ]
        )

        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(
            int(out_channels[0]), features, 3, stride=1, padding=1, bias=False
        )
        self.scratch.layer2_rn = nn.Conv2d(
            int(out_channels[1]), features, 3, stride=1, padding=1, bias=False
        )
        self.scratch.layer3_rn = nn.Conv2d(
            int(out_channels[2]), features, 3, stride=1, padding=1, bias=False
        )
        self.scratch.layer4_rn = nn.Conv2d(
            int(out_channels[3]), features, 3, stride=1, padding=1, bias=False
        )
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(
            features, has_residual=False
        )
        self.proj = nn.Conv2d(
            features,
            self.final_shuffle_factor**2,
            kernel_size=1,
        )

    @staticmethod
    def _validate_features(
        selected_features: Sequence[torch.Tensor],
        student_grid_hw: tuple[int, int],
    ) -> tuple[int, int, int]:
        if len(selected_features) != 4:
            raise ValueError(
                "GeoVRDepthHead expects exactly four selected Qwen feature tensors"
            )
        grid_h, grid_w = (int(value) for value in student_grid_hw)
        if min(grid_h, grid_w) <= 0:
            raise ValueError("The student spatial grid must be positive")
        reference_shape = selected_features[0].shape
        if len(reference_shape) != 3:
            raise ValueError(
                "Each selected feature must have shape [batch, tokens, channels]"
            )
        if reference_shape[1] != grid_h * grid_w:
            raise ValueError(
                "Selected Qwen token count does not match the student grid: "
                f"tokens={reference_shape[1]}, grid={grid_h}x{grid_w}"
            )
        for feature_index, feature in enumerate(selected_features):
            if feature.shape != reference_shape:
                raise ValueError(
                    "All four selected Qwen features must have identical shapes; "
                    f"feature 0={tuple(reference_shape)}, feature "
                    f"{feature_index}={tuple(feature.shape)}"
                )
        return int(reference_shape[0]), grid_h, grid_w

    @staticmethod
    def _apply_position_embedding(
        tensor: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        ratio: float = 0.1,
    ) -> torch.Tensor:
        image_h, image_w = (int(value) for value in image_hw)
        grid_h, grid_w = tensor.shape[-2:]
        position_grid = create_normalized_uv_grid(
            grid_w,
            grid_h,
            aspect_ratio=image_w / image_h,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        position_embedding = position_grid_to_embedding(
            position_grid, tensor.shape[1]
        ).to(dtype=tensor.dtype)
        position_embedding = (
            position_embedding.permute(2, 0, 1)
            .unsqueeze(0)
            .expand(tensor.shape[0], -1, -1, -1)
        )
        return tensor + float(ratio) * position_embedding

    def _fuse(self, feature_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = feature_maps
        layer_1 = self.scratch.layer1_rn(layer_1)
        layer_2 = self.scratch.layer2_rn(layer_2)
        layer_3 = self.scratch.layer3_rn(layer_3)
        layer_4 = self.scratch.layer4_rn(layer_4)
        output = self.scratch.refinenet4(
            layer_4, size=tuple(layer_3.shape[-2:])
        )
        output = self.scratch.refinenet3(
            output, layer_3, size=tuple(layer_2.shape[-2:])
        )
        output = self.scratch.refinenet2(
            output, layer_2, size=tuple(layer_1.shape[-2:])
        )
        return self.scratch.refinenet1(
            output, layer_1, size=tuple(layer_1.shape[-2:])
        )

    def forward(
        self,
        selected_features: Sequence[torch.Tensor],
        *,
        student_grid_hw: tuple[int, int],
        image_hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Predict a positive dense map at the VGGT target resolution."""
        batch_size, grid_h, grid_w = self._validate_features(
            selected_features, student_grid_hw
        )
        image_h, image_w = (int(value) for value in image_hw)
        target_h, target_w = (int(value) for value in target_hw)
        if (image_h, image_w) != (
            grid_h * self.patch_size,
            grid_w * self.patch_size,
        ):
            raise ValueError(
                "Qwen image dimensions do not match its merged visual-token grid: "
                f"image={image_h}x{image_w}, grid={grid_h}x{grid_w}, "
                f"merged_patch={self.patch_size}"
            )

        multi_scale_features = []
        # Relative indexing is intentional: callers provide [H7, H16, H24, H32].
        for feature_index, feature in enumerate(selected_features):
            tensor = self.norm(feature)
            tensor = tensor.transpose(1, 2).reshape(
                batch_size, tensor.shape[-1], grid_h, grid_w
            )
            tensor = self.projects[feature_index](tensor)
            tensor = self._apply_position_embedding(
                tensor, image_hw=(image_h, image_w)
            )
            tensor = self.resize_layers[feature_index](tensor)
            multi_scale_features.append(tensor)

        fused = self._fuse(multi_scale_features)
        # GeoVR's released resolutions are multiples of target_patch_size. The
        # exact Qwen3.5 field of view need not be divisible by the final shuffle
        # factor, so decode the smallest covering canvas and crop after GeoVR's
        # positivity activation.
        desired_fused_hw = (
            math.ceil(target_h / self.final_shuffle_factor),
            math.ceil(target_w / self.final_shuffle_factor),
        )
        if tuple(fused.shape[-2:]) != desired_fused_hw:
            fused = custom_interpolate(
                fused,
                size=desired_fused_hw,
                mode="bilinear",
                align_corners=True,
            )
        fused = self._apply_position_embedding(
            fused, image_hw=(target_h, target_w)
        )

        # Preserve GeoVR's exact positivity order: logits -> shuffle -> BHWC -> exp.
        depth_logits = self.proj(fused)
        depth_logits = F.pixel_shuffle(depth_logits, self.final_shuffle_factor)
        depth_logits = depth_logits.permute(0, 2, 3, 1)
        predicted_depth = torch.exp(depth_logits).squeeze(-1)
        predicted_depth = predicted_depth[..., :target_h, :target_w]
        if predicted_depth.shape != (batch_size, target_h, target_w):
            raise AssertionError(
                "Student and VGGT depth shapes do not match: "
                f"student={tuple(predicted_depth.shape)}, "
                f"target={(batch_size, target_h, target_w)}"
            )
        if not torch.isfinite(predicted_depth).all():
            raise FloatingPointError("Predicted depth contains NaN or Inf")
        if not (predicted_depth > 0).all():
            raise FloatingPointError("GeoVR depth prediction must be strictly positive")
        return predicted_depth


def filter_by_quantile(
    loss_tensor: torch.Tensor,
    valid_range: float,
    min_elements: int = 1000,
    hard_max: float = 100.0,
) -> torch.Tensor:
    """Port of GeoVR's residual-quantile filter, including edge behavior."""
    if loss_tensor.numel() <= min_elements:
        return loss_tensor

    loss_value = loss_tensor.detach().clamp(max=hard_max)
    if loss_value.numel() > 10_000_000:
        permutation = torch.randperm(
            loss_value.numel(), device=loss_value.device
        )[:1_000_000]
        loss_value = loss_value.reshape(-1)[permutation]

    k = math.ceil(float(valid_range) * (loss_value.numel() - 1)) + 1
    quantile_threshold = torch.kthvalue(loss_value.reshape(-1), k)[0]
    quantile_threshold = min(quantile_threshold.item(), float(hard_max))
    mask = loss_tensor <= quantile_threshold
    if mask.sum() < min_elements:
        return loss_tensor.clamp(max=hard_max)
    return loss_tensor[mask]


@dataclass
class DepthLossOutput:
    loss: torch.Tensor
    regression: torch.Tensor
    gradient: torch.Tensor
    mae: torch.Tensor
    valid_fraction: torch.Tensor


def compute_geovr_depth_loss(
    predicted_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    *,
    gradient_scales: Sequence[int] = (1, 2, 4, 8),
    outlier_keep_ratio: float = 0.98,
) -> DepthLossOutput:
    """Compute GeoVR's robust L1 plus strided multi-scale gradient loss."""
    if predicted_depth.shape != teacher_depth.shape:
        raise ValueError(
            "Predicted and teacher depth maps must have identical shapes: "
            f"predicted={tuple(predicted_depth.shape)}, "
            f"teacher={tuple(teacher_depth.shape)}"
        )
    if predicted_depth.ndim < 2:
        raise ValueError("Depth tensors must have at least two spatial dimensions")
    if not 0.0 < float(outlier_keep_ratio) <= 1.0:
        raise ValueError("outlier_keep_ratio must be in (0, 1]")
    scales = tuple(int(scale) for scale in gradient_scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("gradient_scales must contain positive integers")
    if not torch.isfinite(predicted_depth).all():
        raise FloatingPointError("Predicted depth contains NaN or Inf")
    if not (predicted_depth > 0).all():
        raise FloatingPointError("Predicted depth must be strictly positive")

    valid_mask = torch.isfinite(teacher_depth) & (teacher_depth > 0)
    valid_fraction = valid_mask.float().mean()
    if not valid_mask.any():
        connected_zero = predicted_depth.sum() * 0.0
        return DepthLossOutput(
            loss=connected_zero,
            regression=connected_zero,
            gradient=connected_zero,
            mae=connected_zero.detach(),
            valid_fraction=valid_fraction.detach(),
        )

    difference = torch.where(
        valid_mask,
        predicted_depth - teacher_depth,
        torch.zeros_like(predicted_depth),
    )
    absolute_error = difference[valid_mask].abs()
    mae = absolute_error.mean()
    filtered_error = filter_by_quantile(
        absolute_error,
        valid_range=float(outlier_keep_ratio),
    )
    regression_loss = filtered_error.mean()

    gradient_total = predicted_depth.sum() * 0.0
    valid_scale_count = 0
    for step in scales:
        difference_scale = difference[..., ::step, ::step]
        mask_scale = valid_mask[..., ::step, ::step]
        if int(mask_scale.sum().item()) < 10:
            continue
        gradient_x = (
            difference_scale[..., :, 1:] - difference_scale[..., :, :-1]
        ).abs().clamp(max=100.0)
        gradient_y = (
            difference_scale[..., 1:, :] - difference_scale[..., :-1, :]
        ).abs().clamp(max=100.0)
        mask_x = mask_scale[..., :, 1:] & mask_scale[..., :, :-1]
        mask_y = mask_scale[..., 1:, :] & mask_scale[..., :-1, :]
        scale_loss = predicted_depth.sum() * 0.0
        if mask_x.any():
            scale_loss = scale_loss + gradient_x[mask_x].mean()
        if mask_y.any():
            scale_loss = scale_loss + gradient_y[mask_y].mean()
        gradient_total = gradient_total + scale_loss
        valid_scale_count += 1

    gradient_loss = (
        gradient_total / valid_scale_count
        if valid_scale_count
        else predicted_depth.sum() * 0.0
    )
    total_loss = regression_loss + gradient_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Depth loss contains NaN or Inf")
    return DepthLossOutput(
        loss=total_loss,
        regression=regression_loss,
        gradient=gradient_loss,
        mae=mae,
        valid_fraction=valid_fraction.detach(),
    )


def finite_depth_statistics(depth: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return scalar distribution diagnostics over finite depth values."""
    finite = depth.detach().float()[torch.isfinite(depth.detach())]
    device = depth.device
    if finite.numel() == 0:
        nan = torch.tensor(float("nan"), device=device)
        return {
            name: nan
            for name in ("min", "max", "mean", "median", "p01", "p05", "p50", "p95", "p99")
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=finite.device),
    )
    return {
        "min": finite.min(),
        "max": finite.max(),
        "mean": finite.mean(),
        "median": finite.median(),
        "p01": quantiles[0],
        "p05": quantiles[1],
        "p50": quantiles[2],
        "p95": quantiles[3],
        "p99": quantiles[4],
    }
