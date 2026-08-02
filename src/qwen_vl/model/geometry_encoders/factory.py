"""Factory for creating geometry encoders."""

from typing import Optional
from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .vggt_encoder import VGGTEncoder


def create_geometry_encoder(
    encoder_type: str,
    model_path: Optional[str] = None,
    reference_frame: str = "first",
    freeze_encoder: bool = True,
    **encoder_kwargs,
) -> BaseGeometryEncoder:
    """
    Factory function to create geometry encoders.

    Args:
        encoder_type: Must be ``"vggt"``.
        model_path: Path to pretrained model
        reference_frame: Reference frame setting
        freeze_encoder: Whether to freeze encoder parameters
        **encoder_kwargs: Additional encoder-specific arguments

    Returns:
        Geometry encoder instance
    """
    config = GeometryEncoderConfig(
        encoder_type=encoder_type,
        model_path=model_path,
        reference_frame=reference_frame,
        freeze_encoder=freeze_encoder,
        encoder_kwargs=encoder_kwargs,
    )

    if encoder_type != "vggt":
        raise ValueError(f"Unsupported teacher type: {encoder_type}")
    return VGGTEncoder(config)


def get_available_encoders():
    """Get list of available encoder types."""
    return ["vggt"]
