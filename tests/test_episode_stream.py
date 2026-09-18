"""Chronology and exact label accounting, independent of model execution."""
import copy
import pytest
from qwen_vl.contracts import EpisodeRecord, FrameKey, FrameRecord
from qwen_vl.data.episode_stream import EpisodeScheduler


def episodes(actions):
    return tuple(EpisodeRecord(str(i), 'go', 'hash', tuple(
        FrameRecord(FrameKey('r2r', str(i), j), f'{i}/{j}.png', action, f'{i}/{j}')
        for j, action in enumerate(row))) for i, row in enumerate(actions))


def drain(scheduler):
    out = []
    while (schedule := scheduler.plan_update()) is not None:
        out.append(schedule)
    return out


def test_exact_global_labels_chronology_and_idle_ranks():
    data = episodes([['STOP'] * 79, ['STOP'], [None, 'STOP', None], ['STOP'] * 57])
    schedulers = [EpisodeScheduler(data, rank=r, world_size=4, shuffle=False) for r in range(4)]
    plans = [drain(s) for s in schedulers]
    assert [p.global_labels for p in plans[0]] == [64, 64, 10]
    assert all([p.segments_by_rank for p in plans[0]] == [p.segments_by_rank for p in others] for others in plans)
    seen = []
    for plan in plans[0]:
        actual_labels = actual_observations = 0
        for rank, segments in enumerate(plan.segments_by_rank):
            for segment in segments:
                assert segment.episode_index % 4 == rank
                assert 1 <= len(segment.steps) <= 8
                assert segment.first == (segment.steps[0] == 0)
                assert segment.last == (segment.steps[-1] == len(data[segment.episode_index].frames) - 1)
                assert segment.steps == tuple(range(segment.steps[0], segment.steps[-1] + 1))
                seen.extend((segment.episode_index, step) for step in segment.steps)
                actual_observations += len(segment.steps)
                actual_labels += sum(data[segment.episode_index].frames[t].action is not None for t in segment.steps)
        assert (plan.global_labels, plan.global_observations) == (actual_labels, actual_observations)
    assert sorted(seen) == [(i, t) for i, ep in enumerate(data) for t in range(len(ep.frames))]
    assert plans[0][-1].segments_by_rank[1] == ()


def test_unlabeled_terminal_schedule_and_empty_rank():
    data = episodes([['STOP'] * 64 + [None] * 19])
    scheduler = EpisodeScheduler(data, world_size=2, rank=1, shuffle=False)
    first, tail = drain(scheduler)
    assert (first.global_labels, first.global_observations) == (64, 64)
    assert (tail.global_labels, tail.global_observations) == (0, 19)
    assert first.local_segments == tail.local_segments == ()
    assert [len(s.steps) for s in tail.segments_by_rank[0]] == [8, 8, 3]


def test_resume_reproduces_every_remaining_schedule_and_rejects_changes():
    data = episodes([['STOP', None] * n for n in range(1, 13)])
    original = EpisodeScheduler(data, seed=99, rank=1, world_size=3, K=3, target_budget=11)
    original.plan_update()
    state = copy.deepcopy(original.state_dict())
    resumed = EpisodeScheduler(data, seed=99, rank=1, world_size=3, K=3, target_budget=11)
    resumed.load_state_dict(state)
    assert drain(resumed) == drain(original)
    for change in ({'world_size': 2}, {'K': 4}, {'target_budget': 12}):
        args = dict(seed=99, rank=1, world_size=3, K=3, target_budget=11)
        args.update(change)
        with pytest.raises(ValueError):
            EpisodeScheduler(data, **args).load_state_dict(state)
    modified = episodes([['STOP'] * n for n in range(1, 13)])
    with pytest.raises(ValueError):
        EpisodeScheduler(modified, seed=99, rank=1, world_size=3, K=3, target_budget=11).load_state_dict(state)
    state['permutation'] = [0] * len(data)
    with pytest.raises(ValueError):
        resumed.load_state_dict(state)


def test_episode_identity_changes_break_segments():
    plans = drain(EpisodeScheduler(episodes([['STOP']] * 3), shuffle=False, K=8))
    assert len(plans) == 1
    assert all(s.first and s.last and s.steps == (0,) for s in plans[0].local_segments)
