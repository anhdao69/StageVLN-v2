"""Length-aware sampling used by the JanusVLN training recipe."""

from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler
from transformers import Trainer
from transformers.trainer import has_length


_ORIGINAL_GET_TRAIN_SAMPLER = Trainer._get_train_sampler


def _split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks:
        return [indices[index::num_chunks] for index in range(num_chunks)]

    entries_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunk_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        target = chunk_lengths.index(min(chunk_lengths))
        chunks[target].append(index)
        chunk_lengths[target] += lengths[index]
        if len(chunks[target]) == entries_per_chunk:
            chunk_lengths[target] = float("inf")
    return chunks


def _length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """Shuffle length-sorted megabatches and balance them across workers."""
    random_order = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        random_order[offset : offset + megabatch_size].tolist()
        for offset in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda index: lengths[index], reverse=True)
        for megabatch in megabatches
    ]
    balanced = [
        _split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]
    return [index for megabatch in balanced for chunk in megabatch for index in chunk]


class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths,
        generator=None,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        return iter(
            _length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        )


def _get_train_sampler(
    self,
    train_dataset: Optional[Dataset] = None,
) -> Optional[Sampler]:
    train_dataset = train_dataset if train_dataset is not None else self.train_dataset
    if train_dataset is None or not has_length(train_dataset):
        return None
    if getattr(self.args, "group_by_modality_length", False):
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            lengths=train_dataset.modality_lengths,
        )
    return _ORIGINAL_GET_TRAIN_SAMPLER(self, train_dataset)
