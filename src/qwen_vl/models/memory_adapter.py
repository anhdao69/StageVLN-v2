"""Map external memory slots into the native language embedding width."""
import torch
from torch import nn


class MemoryAdapter(nn.Module):
    def __init__(self, text_width: int, width: int = 512):
        super().__init__()
        if min(text_width, width) <= 0:
            raise ValueError('dimensions must be positive')
        self.width = width
        self.norm = nn.LayerNorm(width)
        self.input_projection = nn.Linear(width, 2 * width)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(2 * width, text_width)
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
        self.output_scale = 0.1

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[-1] != self.width:
            raise ValueError('state must have shape [batch, slots, width]')
        hidden = self.activation(self.input_projection(self.norm(state.float())))
        return self.output_projection(hidden) * self.output_scale
