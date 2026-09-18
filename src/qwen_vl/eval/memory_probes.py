"""Explicit, labeled evaluation interventions; never part of default training.

Call step with already observed frames and a callback write(previous)->state.
Use its returned frames/state in the normal reader. A replacement must be a
captured state from another already processed episode, never a future view.
The API reports dependence diagnostics, not navigation quality or causality.
"""
from dataclasses import dataclass
from typing import Callable
import torch
from qwen_vl.contracts import FrameRecord


@dataclass(frozen=True)
class MemorySnapshot:
    episode: str
    step: int
    state: torch.Tensor

    @classmethod
    def capture(cls, episode, step, state):
        if not episode or type(step) is not int or step < 0:
            raise ValueError('Snapshot requires an observed episode/step')
        if state.ndim != 3 or not bool(torch.isfinite(state).all()):
            raise ValueError('Invalid snapshot state')
        return cls(str(episode),step,state.detach().float().clone())


@dataclass(frozen=True)
class ProbeResult:
    state: torch.Tensor
    frames: tuple[FrameRecord,...]
    events: tuple[dict,...]


class MemoryProbe:
    def __init__(self,label,*,reset_steps=(),freeze_after=None,replacements=None,remove_recent=False):
        self.reset_steps=frozenset(reset_steps)
        self.freeze_after=freeze_after
        self.replacements=dict(replacements or {})
        self.remove_recent=bool(remove_recent)
        self.label=label
        if not isinstance(label,str) or not label.strip():
            raise ValueError('Every intervention protocol needs an explicit diagnostic label')
        indices=list(self.reset_steps)+list(self.replacements)
        if freeze_after is not None:indices.append(freeze_after)
        if any(type(step) is not int or step < 0 for step in indices):
            raise ValueError('Intervention steps must be nonnegative integers')
        if self.reset_steps & self.replacements.keys():
            raise ValueError('Reset and replacement at the same step are ambiguous')
        if freeze_after is not None and any(step>freeze_after for step in list(self.reset_steps)+list(self.replacements)):
            raise ValueError('Use separate diagnostics for mutations after freezing')
        if any(not isinstance(snapshot,MemorySnapshot) for snapshot in self.replacements.values()):
            raise TypeError('Replacements must be captured MemorySnapshot values')

    def step(self,episode,step,previous,frames,write:Callable,*,device=None):
        frames=tuple(frames)
        if type(step) is not int or step<0 or not frames or frames[-1].key.step!=step:
            raise ValueError('Probe requires the current observed frame')
        if any(frame.key.episode!=str(episode) or frame.key.step>step for frame in frames):
            raise ValueError('Cross-episode/future observations cannot enter a probe')
        if [f.key.step for f in frames]!=sorted({f.key.step for f in frames}):
            raise ValueError('Observed frames must be unique and chronological')
        events=[]
        def event(kind,**extra):
            events.append(dict(diagnostic=self.label,kind=kind,episode=str(episode),step=step,
                               history_length=step,explicit_history_frames=len(frames)-1,**extra))
        candidate=previous
        if step in self.reset_steps:
            candidate=None;event('reset')
        if step in self.replacements:
            snapshot=self.replacements[step]
            if snapshot.episode==str(episode):
                raise ValueError('Replacement must come from a different observed episode')
            if previous is not None and snapshot.state.shape!=previous.shape:
                raise ValueError('Replacement shape mismatch')
            target_device=device if device is not None else (previous.device if previous is not None else snapshot.state.device)
            candidate=snapshot.state.to(target_device).clone()
            event('replace',source_episode=snapshot.episode,source_step=snapshot.step)
        if self.freeze_after is not None and step>self.freeze_after:
            if candidate is None:raise ValueError('Cannot freeze before a memory state exists')
            state=candidate;event('freeze',after_step=self.freeze_after)
        else:
            state=write(candidate)
        selected=frames
        if self.remove_recent:
            event('remove_recent',cross_protocol=True)
            selected=frames[-1:]
        return ProbeResult(state,selected,tuple(events))
