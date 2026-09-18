"""Deterministic whole-episode assignment and bounded temporal update plans.

Planning advances the scheduler. Save its state only after the corresponding
update completes; keep a pre-plan state if the caller needs update rollback.
Every rank independently produces the same global plan, including idle ranks.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import random

from qwen_vl.contracts import EpisodeRecord


@dataclass(frozen=True)
class Segment:
    episode_index: int
    steps: tuple[int, ...]
    first: bool
    last: bool


@dataclass(frozen=True)
class UpdateSchedule:
    segments_by_rank: tuple[tuple[Segment, ...], ...]
    global_labels: int
    global_observations: int
    rank: int = 0

    @property
    def local_segments(self) -> tuple[Segment, ...]:
        return self.segments_by_rank[self.rank]


class EpisodeScheduler:
    """One stream per rank, K observations per segment, exact global label cap."""

    def __init__(self, episodes: tuple[EpisodeRecord, ...], seed=42, rank=0,
                 world_size=1, K=8, target_budget=64, shuffle=True):
        if world_size < 1 or not 0 <= rank < world_size or K < 1 or target_budget < 1:
            raise ValueError('Invalid rank, world size, K, or target budget')
        self.episodes = tuple(episodes)
        if any(not episode.frames for episode in self.episodes):
            raise ValueError('Empty episodes cannot define a stream')
        if len({episode.episode for episode in self.episodes}) != len(self.episodes):
            raise ValueError('Duplicate episode identity')
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.K, self.target_budget, self.shuffle = K, target_budget, shuffle
        digest = hashlib.sha256()
        for episode in self.episodes:
            digest.update(json.dumps(asdict(episode), sort_keys=True, separators=(',', ':')).encode())
            digest.update(b'\n')
        self.episodes_sha256 = digest.hexdigest()
        self.permutation = list(range(len(self.episodes)))
        if shuffle:
            random.Random(seed).shuffle(self.permutation)
        self._assignments = [self.permutation[r::world_size] for r in range(world_size)]
        self._cursors = [[0, 0] for _ in range(world_size)]
        self._next_rank = 0
        self.total_labels = self.total_observations = 0

    def plan_update(self) -> UpdateSchedule | None:
        segments = [[] for _ in range(self.world_size)]
        labels = observations = 0
        idle = 0
        while labels < self.target_budget and idle < self.world_size:
            rank = self._next_rank
            self._next_rank = (rank + 1) % self.world_size
            episode_position, step = self._cursors[rank]
            if episode_position == len(self._assignments[rank]):
                idle += 1
                continue
            idle = 0
            episode_index = self._assignments[rank][episode_position]
            frames = self.episodes[episode_index].frames
            start = step
            while step < len(frames) and step - start < self.K and labels < self.target_budget:
                labels += int(frames[step].action is not None)
                step += 1
            segments[rank].append(Segment(episode_index, tuple(range(start, step)),
                                          start == 0, step == len(frames)))
            observations += step - start
            self._cursors[rank] = [episode_position + 1, 0] if step == len(frames) else [episode_position, step]
        if not observations:
            return None
        self.total_labels += labels
        self.total_observations += observations
        return UpdateSchedule(tuple(tuple(part) for part in segments), labels, observations, self.rank)

    def state_dict(self):
        return {
            'version': 1, 'episodes_sha256': self.episodes_sha256,
            'world_size': self.world_size, 'K': self.K, 'target_budget': self.target_budget,
            'seed': self.seed, 'shuffle': self.shuffle, 'permutation': list(self.permutation),
            'cursors': [list(cursor) for cursor in self._cursors], 'next_rank': self._next_rank,
            'total_labels': self.total_labels, 'total_observations': self.total_observations,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for name in ('version', 'episodes_sha256', 'world_size', 'K', 'target_budget', 'seed', 'shuffle'):
            if state.get(name) != expected[name]:
                raise ValueError(f'Incompatible episode scheduler {name}')
        permutation = list(state['permutation'])
        if sorted(permutation) != list(range(len(self.episodes))):
            raise ValueError('Invalid episode permutation')
        assignments = [permutation[r::self.world_size] for r in range(self.world_size)]
        cursors = [list(cursor) for cursor in state['cursors']]
        if len(cursors) != self.world_size or not 0 <= state['next_rank'] < self.world_size:
            raise ValueError('Invalid rank cursors')
        observations = labels = 0
        for assignment, cursor in zip(assignments, cursors):
            if len(cursor) != 2 or any(type(x) is not int for x in cursor):
                raise ValueError('Invalid stream cursor')
            position, step = cursor
            if not 0 <= position <= len(assignment):
                raise ValueError('Invalid episode cursor')
            if (position == len(assignment) and step != 0) or (position < len(assignment) and not 0 <= step < len(self.episodes[assignment[position]].frames)):
                raise ValueError('Invalid observation cursor')
            for index in assignment[:position]:
                frames = self.episodes[index].frames
                observations += len(frames)
                labels += sum(frame.action is not None for frame in frames)
            if position < len(assignment):
                frames = self.episodes[assignment[position]].frames[:step]
                observations += len(frames)
                labels += sum(frame.action is not None for frame in frames)
        if (labels, observations) != (state['total_labels'], state['total_observations']):
            raise ValueError('Cursor and consumption counts disagree')
        self.permutation, self._assignments, self._cursors = permutation, assignments, cursors
        self._next_rank = state['next_rank']
        self.total_labels, self.total_observations = labels, observations
