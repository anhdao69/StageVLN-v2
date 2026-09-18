import pytest
import torch
from qwen_vl.contracts import FrameRecord,FrameKey
from qwen_vl.eval.memory_probes import MemoryProbe,MemorySnapshot


def frames(t):
    return tuple(FrameRecord(FrameKey('r2r','a',i),'',None,'') for i in range(max(0,t-4),t+1))

def test_reset_and_freeze_are_explicit_and_causal():
    calls=[]
    def write(previous):
        calls.append(previous)
        return torch.ones(1,2,3) if previous is None else previous+1
    reset=MemoryProbe('reset-at-2',reset_steps={2})
    state=torch.full((1,2,3),5.)
    result=reset.step('a',2,state,frames(2),write)
    assert result.state.eq(1).all() and calls[-1] is None
    assert result.events[0]['history_length']==2
    frozen=MemoryProbe('freeze-after-2',freeze_after=2)
    first=frozen.step('a',2,state,frames(2),write)
    count=len(calls)
    second=frozen.step('a',3,first.state,frames(3),write)
    assert len(calls)==count and torch.equal(first.state,second.state)

def test_cross_episode_replacement_and_window_removal_are_labeled():
    source=torch.full((1,2,3),7.)
    snapshot=MemorySnapshot.capture('b',4,source)
    source.zero_()
    probe=MemoryProbe('cross-episode-control',replacements={3:snapshot},remove_recent=True)
    result=probe.step('a',3,torch.zeros(1,2,3),frames(3),lambda m:m+1)
    assert result.state.eq(8).all() and len(result.frames)==1
    assert snapshot.state.eq(7).all()
    assert {e['kind'] for e in result.events}=={'replace','remove_recent'}
    assert all(e['diagnostic']=='cross-episode-control' for e in result.events)
    with pytest.raises(ValueError):MemoryProbe('',remove_recent=True)
    same=MemoryProbe('bad',replacements={3:MemorySnapshot.capture('a',1,source)})
    with pytest.raises(ValueError):same.step('a',3,source,frames(3),lambda m:m)
    with pytest.raises(ValueError):probe.step('a',2,source,frames(3),lambda m:m)
