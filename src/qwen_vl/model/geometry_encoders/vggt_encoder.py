"""VGGT geometry encoder implementation."""

from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn.functional as F

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class VGGTEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder wrapper."""

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        self.enable_depth = bool(config.encoder_kwargs.get("enable_depth", False))

        # Lazy import to avoid circular dependencies
        from ..vggt.models.vggt import VGGT

        # Initialize VGGT model
        self.vggt = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=self.enable_depth,
            enable_track=False,
        )

        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame
        self.patch_size = 14

    @staticmethod
    def _feature_dtype(images: torch.Tensor) -> torch.dtype:
        if not images.is_cuda:
            return images.dtype
        return (
            torch.bfloat16
            if torch.cuda.get_device_capability(images.device)[0] >= 8
            else torch.float16
        )

    @classmethod
    def _autocast_context(cls, images: torch.Tensor):
        if not images.is_cuda:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=cls._feature_dtype(images))

    def _extract_aggregated_layers(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        images: torch.Tensor,
        *,
        layer_indices: Optional[List[int]],
        spatial_merge_size: int,
        include_camera_token: bool,
    ) -> List[torch.Tensor]:
        """Apply the established v0 token slicing to aggregator outputs."""
        n_image, _, height, width = images.shape
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        spatial_merge_size = (
            spatial_merge_size if spatial_merge_size and spatial_merge_size > 0 else 2
        )
        if layer_indices is None:
            layer_indices = [-2]
        output_dtype = self._feature_dtype(images)

        tensor_features = []
        for index in layer_indices:
            tokens = aggregated_tokens_list[index][0]
            tokens = self._apply_inverse_reference_frame_transform(tokens)
            patch_tokens = tokens[:, patch_start_idx:]
            camera_token = tokens[:, 0:1]

            patch_grid = patch_tokens.reshape(n_image, h_patch, w_patch, -1)
            trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
            trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
            patch_grid = patch_grid[:, :trimmed_h, :trimmed_w, :]
            patch_grid = patch_grid.reshape(
                n_image,
                trimmed_h // spatial_merge_size,
                spatial_merge_size,
                trimmed_w // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            patch_grid = patch_grid.permute(0, 1, 3, 2, 4, 5)
            patch_tokens = patch_grid.reshape(n_image, trimmed_h * trimmed_w, -1)
            geometry_feature = (
                torch.cat((camera_token, patch_tokens), dim=1)
                if include_camera_token
                else patch_tokens
            )
            tensor_features.append(geometry_feature.to(output_dtype).contiguous())
        return tensor_features

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using VGGT and return the default (final) feature set."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)

        with torch.no_grad():
            with self._autocast_context(images):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(
                    images[None]
                )
                features = aggregated_tokens_list[-2][0, :, patch_start_idx:]

        # Apply inverse reference frame transformation
        features = self._apply_inverse_reference_frame_transform(features)

        return features

    def encode_layers(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ):
        """Encode images and return features from specific aggregator layers."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)

        with torch.no_grad():
            with self._autocast_context(images):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(
                    images[None]
                )
        return self._extract_aggregated_layers(
            aggregated_tokens_list,
            patch_start_idx,
            images,
            layer_indices=layer_indices,
            spatial_merge_size=spatial_merge_size,
            include_camera_token=include_camera_token,
        )

    def encode_features_and_depth(
        self,
        images: torch.Tensor,
        *,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ) -> dict[str, torch.Tensor | List[torch.Tensor] | None]:
        """Return v0 features and pseudo-depth from one aggregator execution."""
        if not self.enable_depth or self.vggt.depth_head is None:
            raise RuntimeError("VGGT depth prediction was not enabled")
        if images.ndim != 4:
            raise ValueError(
                f"VGGT images must have shape [frames, 3, H, W], got {tuple(images.shape)}"
            )
        self.vggt.eval()
        transformed_images = self._apply_reference_frame_transform(images)
        original_height, original_width = transformed_images.shape[-2:]
        pad_height = (-original_height) % self.patch_size
        pad_width = (-original_width) % self.patch_size
        if pad_height or pad_width:
            # Qwen3.5 uses 16 px patches, whereas VGGT requires 14 px
            # divisibility. Preserve every Qwen pixel and add only a white
            # bottom/right border for the frozen teacher; dense outputs are
            # cropped back to the exact Qwen field of view below.
            teacher_images = F.pad(
                transformed_images,
                (0, pad_width, 0, pad_height),
                mode="constant",
                value=1.0,
            )
        else:
            teacher_images = transformed_images
        with torch.no_grad():
            with self._autocast_context(teacher_images):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(
                    teacher_images[None]
                )
                spatial_features = self._extract_aggregated_layers(
                    aggregated_tokens_list,
                    patch_start_idx,
                    teacher_images,
                    layer_indices=layer_indices,
                    spatial_merge_size=spatial_merge_size,
                    include_camera_token=include_camera_token,
                )
                depth, depth_confidence = self.vggt.depth_head(
                    aggregated_tokens_list,
                    images=teacher_images[None],
                    patch_start_idx=patch_start_idx,
                )

        depth = depth[..., :original_height, :original_width, :]
        depth_confidence = depth_confidence[
            ..., :original_height, :original_width
        ]
        depth = self._apply_inverse_reference_frame_transform(depth[0])
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        depth_confidence = self._apply_inverse_reference_frame_transform(
            depth_confidence[0]
        )
        if depth.ndim != 3 or depth_confidence.ndim != 3:
            raise AssertionError(
                "VGGT depth outputs must have shape [frames, H, W], got "
                f"depth={tuple(depth.shape)}, confidence={tuple(depth_confidence.shape)}"
            )
        return {
            "sf_features": [feature.detach() for feature in spatial_features],
            "depth": depth.detach(),
            "depth_conf": depth_confidence.detach(),
            "teacher_grid_hw": (
                int(teacher_images.shape[-2]) // self.patch_size,
                int(teacher_images.shape[-1]) // self.patch_size,
            ),
            "teacher_image_hw": (
                int(teacher_images.shape[-2]),
                int(teacher_images.shape[-1]),
            ),
            "teacher_padding_hw": (int(pad_height), int(pad_width)),
        }

    def get_feature_dim(self) -> int:
        """Get VGGT feature dimension."""
        return 2048  # VGGT feature dimension

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)

    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Apply reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images

    def _apply_inverse_reference_frame_transform(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Apply inverse reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    def load_model(self, model_path: str, cache_dir: Optional[str] = None) -> None:
        """Load pretrained VGGT model."""
        from ..vggt.models.vggt import VGGT

        self.vggt = VGGT.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            enable_camera=False,
            enable_point=False,
            enable_depth=self.enable_depth,
            enable_track=False,
        )

        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False
