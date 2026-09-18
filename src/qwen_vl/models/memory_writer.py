"""Pure, explicitly carried external memory with caller-controlled BPTT."""
import torch
from torch import nn

from qwen_vl.contracts import WriterStep


class _WriterBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width))
        self.previous_gate_norm = nn.LayerNorm(width)
        self.proposal_gate_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(2 * width, width)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2)

    def forward(self, previous: torch.Tensor, context: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # The state/residual path stays FP32 even while autocast lowers matmuls.
        previous = previous.float()
        normalized = self.self_norm(previous)
        attended = self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        a = previous + attended.float()
        attended = self.cross_attention(self.cross_norm(a), context, context,
                                        key_padding_mask=~valid_mask, need_weights=False)[0]
        b = a + attended.float()
        proposal = b + self.ffn(self.ffn_norm(b)).float()
        gate_input = torch.cat((self.previous_gate_norm(previous), self.proposal_gate_norm(proposal)), dim=-1)
        gate = torch.sigmoid(self.gate(gate_input)).float()
        return previous + gate * (proposal - previous)


class MemoryWriter(nn.Module):
    """Update memory once per observation; retain only the returned final level.

    Valid masks use True for usable tokens. Nothing is detached implicitly, and
    no episode tensors are stored on the module. A training segment boundary is
    an explicit ``step.state.detach()`` in the caller.
    """
    def __init__(self, slots: int = 64, width: int = 512, layers: int = 3,
                 heads: int = 8, ffn_width: int = 2048):
        super().__init__()
        if min(slots, width, layers, heads, ffn_width) <= 0 or width % heads:
            raise ValueError('positive dimensions and width divisible by heads are required')
        self.slots = slots
        self.width = width
        self.initial_slots = nn.Parameter(torch.empty(slots, width))
        nn.init.normal_(self.initial_slots, mean=0, std=0.02)
        self.blocks = nn.ModuleList([_WriterBlock(width, heads, ffn_width) for _ in range(layers)])

    def initial(self, batch: int, device: torch.device | str) -> torch.Tensor:
        if batch <= 0:
            raise ValueError('batch must be positive')
        return self.initial_slots.to(device=device, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1).clone()

    def reset(self, state: torch.Tensor, first_mask: torch.Tensor) -> torch.Tensor:
        """Reset selected streams; preserve the graph in all continuing streams."""
        self._check_state(state)
        if first_mask.shape != (state.shape[0],) or first_mask.dtype != torch.bool:
            raise ValueError('first_mask must be a boolean batch vector')
        initial = self.initial(state.shape[0], state.device)
        return torch.where(first_mask.to(state.device)[:, None, None], initial, state.float())

    def _check_state(self, state: torch.Tensor) -> None:
        if state.ndim != 3 or state.shape[1:] != (self.slots, self.width):
            raise ValueError('state must have shape [batch, slots, width]')

    def forward(self, previous: torch.Tensor, visual_tokens: torch.Tensor,
                instruction_tokens: torch.Tensor, visual_mask: torch.Tensor,
                instruction_mask: torch.Tensor) -> WriterStep:
        self._check_state(previous)
        for name, tokens, mask in (('visual', visual_tokens, visual_mask),
                                   ('instruction', instruction_tokens, instruction_mask)):
            if tokens.ndim != 3 or tokens.shape[0] != previous.shape[0] or tokens.shape[-1] != self.width:
                raise ValueError(f'{name} tokens must have shape [batch, tokens, width]')
            if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(f'{name} mask must be boolean [batch, tokens]')
            if tokens.device != previous.device or mask.device != previous.device:
                raise ValueError('state, context and masks must be on the same device')
        valid_mask = torch.cat((visual_mask, instruction_mask), dim=1)
        if not valid_mask.any(dim=1).all():
            raise ValueError('each stream requires at least one valid context token')
        context = torch.cat((visual_tokens.float(), instruction_tokens.float()), dim=1)
        # Mask before attention as well as in attention: NaN padding must never
        # contaminate projected values or the QK matmul.
        context = context.masked_fill(~valid_mask[..., None], 0)
        levels = []
        state = previous.float()
        for block in self.blocks:
            state = block(state, context, valid_mask)
            levels.append(state)
        return WriterStep(state=state, levels=tuple(levels))
