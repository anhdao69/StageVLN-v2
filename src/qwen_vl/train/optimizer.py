"""Unique parameter ownership, declared learning rates and optional Adam sharding."""
import torch
import torch.distributed as dist

def parameter_groups(policy,lr_language=1e-6,lr_merger=1e-5,lr_memory=1e-4,weight_decay=0.01):
    groups={};seen=set()
    for name,p in policy.named_parameters():
        if not p.requires_grad:continue
        if id(p) in seen:raise ValueError('Duplicate trainable parameter')
        seen.add(id(p))
        if '.visual.merger.' in name:lr=lr_merger;kind='merger'
        elif name.startswith('backbone.'):lr=lr_language;kind='language'
        else:lr=lr_memory;kind='memory'
        decay=weight_decay if p.ndim>1 and 'norm' not in name.lower() and not name.endswith('bias') else 0.
        group=groups.setdefault((kind,decay),dict(params=[],lr=lr,weight_decay=decay,name=kind))
        group['params'].append(p)
    if not seen:raise ValueError('No trainable parameters')
    return list(groups.values())

def build_optimizer(policy,sharded=False,**kwargs):
    groups=parameter_groups(policy,**kwargs)
    options=dict(lr=kwargs.get('lr_language',1e-6),betas=(0.9,0.999),eps=1e-8)
    if policy.device.type=='cuda':options['fused']=True
    if sharded and dist.is_initialized() and dist.get_world_size()>1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        return ZeroRedundancyOptimizer(groups,optimizer_class=torch.optim.AdamW,**options)
    return torch.optim.AdamW(groups,**options)
