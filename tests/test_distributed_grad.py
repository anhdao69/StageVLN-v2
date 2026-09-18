"""Manual reducer agrees with a global loss, including idle/unused branches."""
from datetime import timedelta
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from qwen_vl.train.distributed_grad import finite_gradients, synchronize_gradients


def _worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    original_all_reduce = dist.all_reduce
    bucket_sizes = []

    def bounded_all_reduce(tensor, *args, **kwargs):
        if tensor.is_floating_point():
            assert tensor.numel() <= 4
            bucket_sizes.append(tensor.numel())
        return original_all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = bounded_all_reduce
    try:
        for counts in ((5, 2), (3, 0)):
            p = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 19))
            branch = torch.nn.Parameter(torch.tensor([2.0]))
            unused = torch.nn.Parameter(torch.ones(2))
            global_count = sum(counts)
            for offset in range(0, counts[rank], 2):
                loss = sum(((p * (1 + rank + j)).square().mean() +
                            (branch.square().sum() if rank == 0 else 0))
                           for j in range(offset, min(offset + 2, counts[rank])))
                (loss * 2 / global_count).backward()
            synchronize_gradients([p, branch, p, unused], bucket_numel=4)
            expected = p.detach().clone().requires_grad_()
            expected_branch = branch.detach().clone().requires_grad_()
            ref = sum((expected * (1 + r + j)).square().mean() +
                      (expected_branch.square().sum() if r == 0 else 0)
                      for r, n in enumerate(counts) for j in range(n)) / global_count
            ref.backward()
            torch.testing.assert_close(p.grad, expected.grad)
            torch.testing.assert_close(branch.grad, expected_branch.grad)
            assert unused.grad is None
            assert finite_gradients([p, branch, unused])
        assert len(bucket_sizes) >= 10  # The 19-value parameter must be sliced.
        if rank == 1:
            p.grad[0] = float('nan')
        assert not finite_gradients([p, branch, unused])
    finally:
        dist.all_reduce = original_all_reduce
        dist.destroy_process_group()


def test_two_rank_unequal_and_empty_gradients_match_global_reference(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)


def test_single_rank_identity_and_validation():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p.square().sum().backward()
    synchronize_gradients([p, p], bucket_numel=1)
    torch.testing.assert_close(p.grad, torch.tensor([2.0, 4.0]))
    assert finite_gradients([p])
    with pytest.raises(ValueError):
        synchronize_gradients([p], bucket_numel=0)
