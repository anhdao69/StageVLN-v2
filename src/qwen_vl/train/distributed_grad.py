"""Single manual FP32 gradient synchronization pass; no DDP/DeepSpeed hooks."""
import torch
import torch.distributed as dist


def _parameters(parameters):
    result, seen = [], set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _device(parameters):
    if _distributed() and dist.get_backend() == 'nccl':
        if not parameters or parameters[0].device.type != 'cuda':
            raise ValueError('NCCL requires CUDA parameters')
        return parameters[0].device
    return torch.device('cpu')


@torch.no_grad()
def synchronize_gradients(parameters, bucket_numel=16_000_000):
    """SUM then divide by world size, with bounded FP32 communication storage.

    The caller supplies the same ordered parameters on all ranks and scales its
    local loss by world_size/global_labels. Duplicate identities are ignored.
    Globally unused gradients remain None. Large parameters are split across
    buckets; an absent local gradient allocates only its final gradient storage.
    """
    if type(bucket_numel) is not int or bucket_numel < 1:
        raise ValueError('bucket_numel must be a positive integer')
    parameters = _parameters(parameters)
    distributed = _distributed()
    world_size = dist.get_world_size() if distributed else 1
    if not parameters:
        return
    device = parameters[0].device
    if any(p.device != device for p in parameters):
        raise ValueError('All parameters must reside on the same device')
    active = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32,
                          device=_device(parameters))
    if distributed:
        dist.all_reduce(active, op=dist.ReduceOp.MAX)
    active = active.cpu().tolist()
    pieces, used = [], 0
    bucket = torch.empty(min(bucket_numel, sum(p.numel() for p, flag in zip(parameters, active) if flag)),
                         dtype=torch.float32, device=device)

    def flush():
        nonlocal used
        if not used:
            return
        values = bucket[:used]
        if distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values.div_(world_size)
        for grad, begin, size, bucket_begin in pieces:
            grad[begin:begin + size].copy_(values[bucket_begin:bucket_begin + size])
        pieces.clear()
        used = 0

    for parameter, flag in zip(parameters, active):
        if not flag:
            parameter.grad = None
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter, memory_format=torch.contiguous_format)
        elif parameter.grad.is_sparse:
            raise ValueError('Sparse gradients are not supported')
        elif not parameter.grad.is_contiguous():
            parameter.grad = parameter.grad.contiguous()
        grad = parameter.grad.view(-1)
        for begin in range(0, grad.numel(), bucket_numel):
            # Split once more at a partially occupied bucket boundary.
            stop = min(begin + bucket_numel, grad.numel())
            while begin < stop:
                size = min(stop - begin, bucket.numel() - used)
                bucket[used:used + size].copy_(grad[begin:begin + size])
                pieces.append((grad, begin, size, used))
                used += size
                begin += size
                if used == bucket.numel():
                    flush()
    flush()


@torch.no_grad()
def finite_gradients(parameters) -> bool:
    """One shared finite decision, including ranks with no local gradients."""
    parameters = _parameters(parameters)
    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    flag = torch.tensor(int(finite), dtype=torch.int32, device=_device(parameters))
    if _distributed():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())
