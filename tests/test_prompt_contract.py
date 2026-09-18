from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer
from qwen_vl.contracts import EpisodeRecord,FrameRecord,FrameKey
from qwen_vl.data.data_qwen import QWEN3_5_NON_THINKING_CHAT_TEMPLATE
from qwen_vl.data.prompting import build_state


@pytest.fixture(scope='module')
def tokenizer():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/datasets/v2_mem64_r0.json').read_text())
    tok=AutoTokenizer.from_pretrained(cfg['model'])
    tok.chat_template=QWEN3_5_NON_THINKING_CHAT_TEMPLATE
    return tok


@pytest.mark.parametrize('count',[1,5,9])
@pytest.mark.parametrize('slots',[0,64])
def test_training_prefix_is_label_independent(tokenizer,count,slots):
    frames=tuple(FrameRecord(FrameKey('r2r','a',i),'unused','TURN_LEFT',str(i)) for i in range(count))
    ep=EpisodeRecord('a','Find the kitchen.','',frames)
    grids=[(1,4,6)]*count
    expected=None
    for action in ('MOVE_FORWARD','TURN_LEFT','TURN_RIGHT','STOP'):
        current=frames[:-1]+(replace(frames[-1],action=action),)
        training=build_state(ep,current,tokenizer,grids,memory_slots=slots)
        inference=build_state(ep,current,tokenizer,grids,memory_slots=slots,include_target=False)
        torch.testing.assert_close(inference.input_ids,training.input_ids[:training.action_start])
        assert bool(inference.labels.eq(-100).all())
        if expected is not None:
            torch.testing.assert_close(inference.input_ids,expected)
        expected=inference.input_ids
        if slots:
            start,stop=training.memory_span
            assert stop-start==64 and bool(training.labels[start:stop].eq(-100).all())
            assert not any(int(i) in tokenizer.all_special_ids for i in training.input_ids[start:stop])
        else:
            assert training.memory_span is None
            assert 'Memory state:' not in tokenizer.decode(training.input_ids)
