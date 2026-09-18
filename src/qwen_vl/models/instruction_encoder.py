"""Compress detached lexical instruction embeddings into learned queries."""
import math

import torch
from torch import nn


def _positions(length: int, width: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32, device=device)[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32, device=device)
                            * (-math.log(10000.0) / width))
    angles = positions * frequencies[None]
    encoding = torch.empty(length, width, dtype=torch.float32, device=device)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, :width // 2].cos()
    return encoding


class InstructionEncoder(nn.Module):
    """One masked text block followed by query cross-attention and FFN.

    Supply actual instruction lexical embeddings only, without prompt/action
    markup. True marks real instruction tokens; each instruction has 1–512.
    The existing language embedding table is intentionally detached here.
    """
    def __init__(self, text_width: int, width: int = 512, heads: int = 8,
                 ffn_width: int = 2048, queries: int = 8):
        super().__init__()
        if min(text_width, width, heads, ffn_width, queries) <= 0 or width % heads:
            raise ValueError('positive dimensions and width divisible by heads are required')
        self.text_width = text_width
        self.width = width
        self.projection = nn.Linear(text_width, width)
        self.text_attention_norm = nn.LayerNorm(width)
        self.text_attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.text_ffn_norm = nn.LayerNorm(width)
        self.text_ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width))
        self.queries = nn.Parameter(torch.empty(queries, width))
        nn.init.normal_(self.queries, std=0.02)
        self.query_attention_norm = nn.LayerNorm(width)
        self.query_attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.query_ffn_norm = nn.LayerNorm(width)
        self.query_ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width))

    def forward(self, detached_embeddings: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if detached_embeddings.ndim != 3 or detached_embeddings.shape[-1] != self.text_width:
            raise ValueError('embeddings must have shape [batch, length, text_width]')
        if valid_mask.shape != detached_embeddings.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError('valid_mask must be boolean [batch, length]')
        if valid_mask.device != detached_embeddings.device:
            raise ValueError('embeddings and mask must be on the same device')
        counts = valid_mask.sum(dim=1)
        if detached_embeddings.shape[1] == 0 or not ((counts > 0) & (counts <= 512)).all():
            raise ValueError('instructions require 1–512 valid lexical tokens')
        lexical = detached_embeddings.detach().float().masked_fill(~valid_mask[..., None], 0)
        encoded = self.projection(lexical).float()
        encoded = encoded + _positions(encoded.shape[1], self.width, encoded.device)[None]
        normalized = self.text_attention_norm(encoded)
        attended = self.text_attention(normalized, normalized, normalized,
                                       key_padding_mask=~valid_mask, need_weights=False)[0]
        encoded = encoded + attended.float()
        encoded = encoded + self.text_ffn(self.text_ffn_norm(encoded)).float()
        encoded = encoded.masked_fill(~valid_mask[..., None], 0)
        query = self.queries.float()[None].expand(encoded.shape[0], -1, -1)
        attended = self.query_attention(self.query_attention_norm(query), encoded, encoded,
                                        key_padding_mask=~valid_mask, need_weights=False)[0]
        query = query + attended.float()
        return query + self.query_ffn(self.query_ffn_norm(query)).float()
