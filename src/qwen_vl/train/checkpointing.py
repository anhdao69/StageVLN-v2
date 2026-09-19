"""Completed-update checkpoints for deterministic fixed-world-size continuation.

Only load trusted local checkpoints: RNG/cursor payloads use torch pickle.
All ranks call save/load with the same shared directory and manifest. Rank zero
writes model weights once; each rank writes optimizer, scheduler, episode plan,
RNG and detached CPU FP32 cursor tensors. An atomic completion manifest is
published only after all payloads exist. Existing completed paths are immutable.
"""
from dataclasses import fields, is_dataclass, replace
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


def _rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def capture_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _cursor_payload(value):
    if isinstance(value, torch.Tensor):
        dtype = torch.float32 if value.is_floating_point() else value.dtype
        return value.detach().to(device='cpu', dtype=dtype).clone()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _cursor_payload(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, dict):
        return {key: _cursor_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cursor_payload(item) for item in value)
    if isinstance(value, list):
        return [_cursor_payload(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported cursor value: {type(value).__name__}')


def _atomic_save(payload, path):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(path, policy, optimizer, scheduler, episode_scheduler,
                    cursors, manifest, step):
    """Save after all backwards, optimizer/scheduler steps and cursor commits.

    `manifest` is an exact JSON-compatible identity/config record; include base
    revision, source SHA256, architecture, preprocessing and prompt hashes.
    `cursors` contains persistent state only, never instruction feature caches.
    Native optimizer wrappers with `.optim` save their local optimizer shard.
    """
    path = Path(path)
    rank, world_size = _rank_world()
    manifest = json.loads(json.dumps(manifest, sort_keys=True))
    if step < 0:
        raise ValueError('Checkpoint step must be nonnegative')
    if episode_scheduler.world_size != world_size:
        raise ValueError('Episode scheduler world size mismatch')
    if path.joinpath('manifest.json').exists():
        raise FileExistsError(f'Completed checkpoint already exists: {path}')
    path.mkdir(parents=True, exist_ok=True)
    local_optimizer = getattr(optimizer, 'optim', optimizer)
    payload = {
        'step': step, 'rank': rank, 'world_size': world_size,
        'optimizer': local_optimizer.state_dict(),
        'optimizer_sharded': local_optimizer is not optimizer,
        'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'}
                             for group in optimizer.param_groups],
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'episode_scheduler': episode_scheduler.state_dict(),
        'cursors': _cursor_payload(cursors), 'rng': capture_rng_state(),
    }
    _atomic_save(payload, path / f'rank_{rank:05d}.pt')
    if rank == 0:
        _atomic_save(policy.state_dict(), path / 'model.pt')
    _barrier()
    if rank == 0:
        completion = {'format_version': 1, 'world_size': world_size, 'step': step,
                      'epoch': episode_scheduler.epoch,
                      'manifest': manifest, 'total_labels': episode_scheduler.total_labels,
                      'total_observations': episode_scheduler.total_observations}
        temporary = path / 'manifest.json.tmp'
        temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + '\n')
        os.replace(temporary, path / 'manifest.json')
    _barrier()


def load_checkpoint(path, policy, optimizer, scheduler, episode_scheduler, *, expected_manifest):
    """Restore in-place, then return step/cursors and consumption counts.

    Cursor tensors are CPU FP32; the trainer explicitly moves live memory to its
    execution device. RNG is restored last, after model/optimizer loading.
    """
    path = Path(path)
    rank, world_size = _rank_world()
    completion = json.loads((path / 'manifest.json').read_text())
    expected_manifest = json.loads(json.dumps(expected_manifest, sort_keys=True))
    if completion.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    if completion['world_size'] != world_size or episode_scheduler.world_size != world_size:
        raise ValueError('Checkpoint world size mismatch; repartition is unsupported')
    if completion['manifest'] != expected_manifest:
        raise ValueError('Checkpoint manifest mismatch')
    payload = torch.load(path / f'rank_{rank:05d}.pt', map_location='cpu', weights_only=False)
    if (payload['rank'], payload['world_size'], payload['step']) != (rank, world_size, completion['step']):
        raise ValueError('Checkpoint rank/step payload mismatch')
    local_optimizer = getattr(optimizer, 'optim', optimizer)
    if payload['optimizer_sharded'] != (local_optimizer is not optimizer):
        raise ValueError('Checkpoint optimizer backend mismatch')
    if (payload['scheduler'] is None) != (scheduler is None):
        raise ValueError('Checkpoint LR scheduler mismatch')
    episode_scheduler.load_state_dict(payload['episode_scheduler'])
    policy.load_state_dict(torch.load(path / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    local_optimizer.load_state_dict(payload['optimizer'])
    # ZeRO copies wrapper hyperparameters into its local optimizer on every
    # step. Restore those too, or the first resumed step resets the saved LR.
    if len(optimizer.param_groups) != len(payload['optimizer_groups']):
        raise ValueError('Checkpoint optimizer parameter groups mismatch')
    for group, saved in zip(optimizer.param_groups, payload['optimizer_groups']):
        group.update(saved)
    if scheduler is not None:
        scheduler.load_state_dict(payload['scheduler'])
    _barrier()
    restore_rng_state(payload['rng'])
    return {'step': payload['step'], 'cursors': payload['cursors'],
            'total_labels': completion['total_labels'],
            'total_observations': completion['total_observations'], 'manifest': completion['manifest']}
