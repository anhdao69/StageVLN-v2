"""Writer-only pooling of native block-major pre-merger image features."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def premerge_to_raster(rows: torch.Tensor, grid_thw: tuple[int, int, int], merge_size: int) -> torch.Tensor:
    """Undo Qwen's [T,H/m,W/m,m,m,D] merge-block row ordering."""
    t, h, w = (int(value) for value in grid_thw)
    m = merge_size
    if (rows.ndim != 2 or m <= 0 or t != 1 or h <= 0 or w <= 0
            or h % m or w % m or rows.shape[0] != t * h * w):
        raise ValueError('unsupported image grid')
    d = rows.shape[-1]
    return rows.reshape(t, h // m, w // m, m, m, d).permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, d)


def _coordinate_encoding(height: int, width: int, channels: int, device: torch.device) -> torch.Tensor:
    # Cell centers in the normalized image, with each axis encoded separately.
    v = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height
    u = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    frequencies = torch.exp(torch.arange(channels // 4, dtype=torch.float32, device=device)
                            * (-math.log(10000.0) / (channels // 4)))
    def axis(values):
        angles = 2 * math.pi * values[..., None] * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)
    return torch.cat((axis(uu), axis(vv)), dim=-1).reshape(1, height * width, channels)


class VisualTokenProjector(nn.Module):
    """Project before pooling; never alter native reader visual tokens.

    One image enters each call. Its pooled cells are all valid, so the returned
    mask is all True; callers padding multiple outputs must extend that mask.
    """
    def __init__(self, visual_width: int, width: int = 512, merge_size: int = 2):
        super().__init__()
        if min(visual_width, width, merge_size) <= 0 or width % 4:
            raise ValueError('positive dimensions and width divisible by four are required')
        self.visual_width = visual_width
        self.width = width
        self.merge_size = merge_size
        self.projection = nn.Linear(visual_width, width)
        self.visual_type = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.visual_type, std=0.02)

    def forward(self, premerge: torch.Tensor, grid_thw: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if premerge.ndim != 2 or premerge.shape[-1] != self.visual_width:
            raise ValueError('premerge must have shape [patches, visual_width]')
        raster = premerge_to_raster(premerge, grid_thw, self.merge_size)
        _, h, w, _ = raster.shape
        h2 = min(h, max(1, round(8 * h / max(h, w))))
        w2 = min(w, max(1, round(8 * w / max(h, w))))
        projected = self.projection(raster.detach().float()).float().permute(0, 3, 1, 2)
        tokens = F.adaptive_avg_pool2d(projected, (h2, w2)).flatten(2).transpose(1, 2)
        tokens = tokens + _coordinate_encoding(h2, w2, self.width, tokens.device) + self.visual_type.float()[None, None]
        mask = torch.ones((1, h2 * w2), dtype=torch.bool, device=tokens.device)
        return tokens, mask
