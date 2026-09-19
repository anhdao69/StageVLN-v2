import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from transformers import get_cosine_schedule_with_warmup

from qwen_vl.data.episode_stream import EpisodeScheduler
from qwen_vl.train.checkpointing import save_checkpoint, load_checkpoint
from qwen_vl.train.retention import prune_steps
from test_episode_stream import episodes, drain


def test_five_epochs_have_separate_tails_and_exact_resume():
    data = episodes([['STOP']*n for n in range(1, 13)])  # 78 labels, 64+14 each epoch
    stream = EpisodeScheduler(data, world_size=4)
    with pytest.raises(ValueError, match='unfinished'):
        stream.advance_epoch()
    permutations = []
    for epoch in range(1, 6):
        assert stream.epoch == epoch
        permutations.append(tuple(stream.permutation))
        first = stream.plan_update()
        restored = EpisodeScheduler(data, world_size=4)
        restored.load_state_dict(copy.deepcopy(stream.state_dict()))
        remaining = drain(stream)
        assert remaining == drain(restored)
        assert [p.global_labels for p in [first]+remaining] == [64, 14]
        seen = [(s.episode_index, t) for p in [first]+remaining for rank in p.segments_by_rank for s in rank for t in s.steps]
        assert sorted(seen) == [(i,t) for i,ep in enumerate(data) for t in range(len(ep.frames))]
        if epoch < 5:
            stream.advance_epoch()
            restored.advance_epoch()
            assert stream.state_dict() == restored.state_dict()
    assert len(set(permutations)) == 5


def test_full_state_resume_at_epoch_boundary_and_next_epoch(tmp_path):
    data = episodes([['STOP']*3])
    def objects():
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        lr = get_cosine_schedule_with_warmup(optimizer, 1, 10)
        stream = EpisodeScheduler(data, target_budget=2)
        return model, optimizer, lr, stream
    def update(objects):
        model, optimizer, lr, stream = objects
        plan = stream.plan_update()
        assert plan is not None
        optimizer.zero_grad()
        model(torch.randn(plan.global_labels, 2)).square().mean().backward()
        optimizer.step(); lr.step()
    original = objects()
    update(original); update(original)
    assert original[3].exhausted
    save_checkpoint(tmp_path/'epoch-1', *original, {}, {'epochs':5}, 2)
    original[3].advance_epoch()
    update(original)
    expected = copy.deepcopy(original[0].state_dict())
    resumed = objects()
    state = load_checkpoint(tmp_path/'epoch-1', *resumed, expected_manifest={'epochs':5})
    assert state['step'] == 2 and resumed[3].epoch == 1 and resumed[3].exhausted
    resumed[3].advance_epoch()
    update(resumed)
    assert resumed[2].state_dict() == original[2].state_dict()
    for key, value in expected.items():
        torch.testing.assert_close(resumed[0].state_dict()[key], value, rtol=0, atol=0)
    save_checkpoint(tmp_path/'step-epoch-2-3', *resumed, {}, {'epochs':5}, 3)
    update(resumed)
    expected = copy.deepcopy(resumed[0].state_dict())
    mid = objects()
    load_checkpoint(tmp_path/'step-epoch-2-3', *mid, expected_manifest={'epochs':5})
    update(mid)
    for key, value in expected.items():
        torch.testing.assert_close(mid[0].state_dict()[key], value, rtol=0, atol=0)
    # Continue through all five epochs without resetting Adam or cosine LR.
    for epoch in range(3, 6):
        mid[3].advance_epoch()
        update(mid); update(mid)
        save_checkpoint(tmp_path/f'epoch-{epoch}', *mid, {}, {'epochs':5}, epoch*2)
    assert mid[2].last_epoch == 10 and mid[2].get_last_lr() == [0.0]
    assert all(int(state['step']) == 10 for state in mid[1].state.values())


def test_retention_preserves_epochs_incomplete_and_other_epochs(tmp_path):
    for name in ['epoch-1', 'step-epoch-1-500'] + [f'step-epoch-2-{i}' for i in [9,10,20,30,40,50,100]]:
        path = tmp_path/name; path.mkdir(); (path/'manifest.json').write_text('{}')
    (tmp_path/'step-epoch-2-200').mkdir()
    prune_steps(tmp_path, 2)
    assert sorted(p.name for p in tmp_path.glob('step-epoch-2-*') if (p/'manifest.json').exists()) == sorted(f'step-epoch-2-{i}' for i in [20,30,40,50,100])
    assert (tmp_path/'epoch-1/manifest.json').exists()
    assert (tmp_path/'step-epoch-1-500/manifest.json').exists()
    assert (tmp_path/'step-epoch-2-200').exists()
    prune_steps(tmp_path, 2, keep=0)
    assert (tmp_path/'epoch-1/manifest.json').exists()


def test_epoch_upload_completion_guard_and_idempotence(tmp_path):
    spec = importlib.util.spec_from_file_location('upload_epochs', Path(__file__).parents[1]/'scripts/recurrent/upload_epochs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    class API:
        calls = 0
        def create_repo(self, *args, **kwargs): assert kwargs['private']
        def upload_folder(self, **kwargs): self.calls += 1
    api = API()
    assert not module.upload_epoch(api, tmp_path, 1, 'owner/run')
    directory = tmp_path/'epoch-1'; export = directory/'export'; (export/'backbone').mkdir(parents=True)
    (directory/'EPOCH_COMPLETE').write_text('{"epoch":1}')
    (export/'run_manifest.json').write_text('{"max_episodes":1}')
    with pytest.raises(ValueError, match='smoke'):
        module.upload_epoch(api, tmp_path, 1, 'owner/run')
    (export/'run_manifest.json').write_text('{"max_episodes":0}')
    for name in ['navigation_config.json','prompt_protocol.json','memory.pt','backbone/model.safetensors']:
        (export/name).write_text('test')
    assert module.upload_epoch(api, tmp_path, 1, 'owner/run')
    assert module.upload_epoch(api, tmp_path, 1, 'owner/run')
    assert api.calls == 1
