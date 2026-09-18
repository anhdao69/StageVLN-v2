"""A completed-update checkpoint reproduces the next stochastic update."""
import random
import json
import numpy as np
import pytest
import torch
from qwen_vl.train.checkpointing import capture_rng_state, restore_rng_state, save_checkpoint, load_checkpoint
from qwen_vl.data.episode_stream import EpisodeScheduler
from qwen_vl.contracts import EpisodeRecord, FrameRecord, FrameKey


def _objects():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    lr = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    data = (EpisodeRecord('0', 'go', 'hash', tuple(
        FrameRecord(FrameKey('r2r', '0', t), f'{t}.png', 'STOP', str(t)) for t in range(8))),)
    return model, optimizer, lr, EpisodeScheduler(data, target_budget=2)


def _update(model, optimizer, lr, episodes):
    schedule = episodes.plan_update()
    x = torch.rand(2, 2) + random.random() + np.random.random()
    loss = model(x).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr.step()
    return loss.detach(), schedule


def test_checkpoint_replays_rng_optimizer_scheduler_and_cursors(tmp_path):
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    model, optimizer, lr, episodes = _objects()
    _update(model, optimizer, lr, episodes)
    memory = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True) * 2
    cursors = {'episode': '0', 'next_step': 2, 'memory': memory, 'instruction_ids': [1, 2]}
    manifest = {'data_sha256': 'abc', 'revision': 'rev', 'prompt_hash': 'prompt'}
    save_checkpoint(tmp_path / 'saved', model, optimizer, lr, episodes, cursors, manifest, 1)
    expected_loss, expected_schedule = _update(model, optimizer, lr, episodes)
    expected_model = {k: v.clone() for k, v in model.state_dict().items()}
    model2, optimizer2, lr2, episodes2 = _objects()
    restored = load_checkpoint(tmp_path / 'saved', model2, optimizer2, lr2, episodes2, expected_manifest=manifest)
    assert restored['step'] == 1
    assert restored['cursors']['memory'].dtype == torch.float32
    assert restored['cursors']['memory'].grad_fn is None
    assert restored['cursors']['next_step'] == 2
    actual_loss, actual_schedule = _update(model2, optimizer2, lr2, episodes2)
    assert actual_schedule == expected_schedule
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for k, value in model2.state_dict().items():
        torch.testing.assert_close(value, expected_model[k], rtol=0, atol=0)
    for old, new in zip(optimizer.state.values(), optimizer2.state.values()):
        for k in old:
            torch.testing.assert_close(old[k], new[k], rtol=0, atol=0)
    assert lr.state_dict() == lr2.state_dict()
    with pytest.raises(ValueError, match='manifest'):
        load_checkpoint(tmp_path / 'saved', model2, optimizer2, lr2, episodes2,
                        expected_manifest={'data_sha256': 'changed'})


def test_rng_roundtrip_and_incomplete_checkpoint(tmp_path):
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))
    assert actual == expected
    with pytest.raises((ValueError, FileNotFoundError)):
        load_checkpoint(tmp_path, *_objects(), expected_manifest={})


def _sharded_checkpoint_worker(rank, rendezvous, directory):
    from datetime import timedelta
    import torch.distributed as dist
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from qwen_vl.train.distributed_grad import synchronize_gradients
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        def objects():
            model, _, _, _ = _objects()
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=0.01)
            lr = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
            data = tuple(EpisodeRecord(str(i), 'go', 'hash', tuple(
                FrameRecord(FrameKey('r2r', str(i), t), f'{i}/{t}.png', 'STOP', f'{i}/{t}')
                for t in range(8))) for i in range(2))
            return model, optimizer, lr, EpisodeScheduler(data, rank=rank, world_size=2, K=1, target_budget=2)

        def update(model, optimizer, lr, episodes):
            schedule = episodes.plan_update()
            x = torch.rand(2, 2) + random.random() + np.random.random()
            loss = model(x).square().mean()
            optimizer.zero_grad()
            loss.backward()
            synchronize_gradients(model.parameters(), bucket_numel=2)
            optimizer.step()
            lr.step()
            return loss.detach(), schedule

        torch.manual_seed(11)
        original = objects()
        random.seed(rank + 7); np.random.seed(rank + 7); torch.manual_seed(rank + 7)
        update(*original)
        save_checkpoint(directory, *original, {'rank': rank, 'memory': torch.tensor([float(rank)])}, {'sha': 'same'}, 1)
        expected_loss, expected_schedule = update(*original)
        restored_objects = objects()
        restored = load_checkpoint(directory, *restored_objects, expected_manifest={'sha': 'same'})
        assert restored['cursors']['rank'] == rank
        actual_loss, actual_schedule = update(*restored_objects)
        assert actual_schedule == expected_schedule
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for p, q in zip(original[0].parameters(), restored_objects[0].parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert original[2].get_last_lr() == restored_objects[2].get_last_lr()
        for old, new in zip(original[1].optim.state.values(), restored_objects[1].optim.state.values()):
            for key in old:
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_sharded_optimizer_resume(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_sharded_checkpoint_worker,
             args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'saved')),
             nprocs=2, join=True)


def test_completion_manifest_rejects_changed_world_and_overwrite(tmp_path):
    objects = _objects()
    path = tmp_path / 'saved'
    save_checkpoint(path, *objects, {}, {'sha': 'same'}, 0)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, *objects, {}, {'sha': 'same'}, 0)
    metadata_path = path / 'manifest.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['world_size'] = 2
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='world size'):
        load_checkpoint(path, *objects, expected_manifest={'sha': 'same'})
