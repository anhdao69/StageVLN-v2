"""R2R chronological training. Launch with torchrun; never wraps the model in DDP.

Trainable weights/gradients/Adam moments are FP32. BF16 autocast, native Adam
state sharding and the tested leaf-gradient bridge reduce memory use without
cutting temporal credit. All ranks participate in one reduction per update.
"""
import argparse
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, get_cosine_schedule_with_warmup

from qwen_vl.data.episode_manifest import load_manifest
from qwen_vl.data.episode_stream import EpisodeScheduler
from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.checkpointing import load_checkpoint, save_checkpoint
from qwen_vl.train.distributed_grad import finite_gradients, synchronize_gradients
from qwen_vl.train.episode_trainer import EpisodeTrainer
from qwen_vl.train.optimizer import build_optimizer
from qwen_vl.train.retention import prune_steps


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-updates', type=int, default=0, help='0 means all configured epochs; otherwise an absolute global step limit')
    parser.add_argument('--max-episodes', type=int, default=0, help='Smoke only: keep complete episodes')
    parser.add_argument('--reader-microbatch', type=int)
    parser.add_argument('--execution', choices=['direct', 'bridge'])
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--resume')
    parser.add_argument('--skip-final-save', action='store_true', help='Bounded smoke runs only')
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    return parser.parse_args()


def final_checkpoint_path(output, step, observations):
    """A zero-label tail can advance a cursor without increasing the step."""
    path = Path(output)/f'checkpoint-{step}'
    completion = path/'manifest.json'
    if completion.exists() and json.loads(completion.read_text())['total_observations'] != observations:
        path = Path(output)/f'checkpoint-{step}-observations-{observations}'
    return path


def implementation_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    args = arguments()
    if min(args.max_updates,args.max_episodes,args.checkpoint_every)<0:
        raise ValueError('Update/episode/checkpoint limits must be nonnegative')
    cfg = json.loads(Path(args.config).read_text())
    epochs = cfg.get('epochs', 1)
    if type(epochs) is not int or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    if args.skip_final_save and not args.max_updates:
        raise ValueError('--skip-final-save is restricted to bounded smoke runs')
    os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1' if cfg['attention_deterministic'] else '0'
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if not torch.cuda.is_available():
        raise RuntimeError('This full-model runner requires CUDA; CPU contracts have separate tests')
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank), timeout=timedelta(hours=2))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    episodes = load_manifest(cfg['episode_manifest'], cfg['episode_manifest_sha256'])
    full_labels = sum(f.action is not None for ep in episodes for f in ep.frames)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    stream = EpisodeScheduler(episodes, seed=cfg['seed'], rank=rank, world_size=world,
                              K=cfg['segment_length'], target_budget=cfg['global_batch_size'])
    total_labels = sum(f.action is not None for ep in episodes for f in ep.frames)
    updates_per_epoch = math.ceil(total_labels / cfg['global_batch_size'])
    total_updates = epochs * updates_per_epoch
    processor = AutoProcessor.from_pretrained(cfg['model'])
    processor.image_processor.size.update(longest_edge=cfg['max_pixels'], shortest_edge=cfg['min_pixels'])
    processor.image_processor.max_pixels = cfg['max_pixels']
    processor.image_processor.min_pixels = cfg['min_pixels']
    from qwen_vl.train.train_qwen import _install_qwen35_flash_attention_fix
    _install_qwen35_flash_attention_fix()
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg['model'], dtype=torch.float32, attn_implementation=cfg['attention'],
        device_map={'': local_rank})
    backbone.model.visual.requires_grad_(False)
    backbone.model.visual.to(dtype=torch.bfloat16)
    backbone.model.visual.merger.to(dtype=torch.float32).requires_grad_(True)
    checkpointing = cfg['gradient_checkpointing'] and not args.no_gradient_checkpointing
    if checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    policy = NavigationPolicy(backbone, processor.tokenizer, cfg['memory_enabled']).to(device=local_rank)
    policy.train()
    optimizer = build_optimizer(policy, sharded=cfg['optimizer_sharding'],
                                lr_language=cfg['lr_language'], lr_merger=cfg['lr_merger'],
                                lr_memory=cfg['lr_memory'], weight_decay=cfg['weight_decay'])
    warmup = cfg['warmup_updates']
    if warmup < 1:
        raise ValueError('warmup_updates must be positive')
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup,
                                               num_training_steps=total_updates)
    loader = FrameLoader(cfg['data_root'], processor.image_processor, workers=cfg['workers_per_rank'])
    execution = args.execution or cfg['execution']
    microbatch = args.reader_microbatch or cfg['reader_microbatch']
    trainer = EpisodeTrainer(policy, episodes, loader, cfg['recent'], execution, microbatch)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in parameters)
    manifest = dict(format_version=1, config=cfg, world_size=world, execution=execution,
                    reader_microbatch=microbatch, gradient_checkpointing=checkpointing,
                    episodes_sha256=stream.episodes_sha256,
                    source_sha256=hashlib.sha256(Path(cfg['episode_manifest']).read_bytes()).hexdigest(),
                    chat_sha256=hashlib.sha256(policy.tokenizer.chat_template.encode()).hexdigest(),
                    implementation_sha256=implementation_fingerprint(),
                    memory_architecture=policy.memory_config if policy.memory_enabled else None,
                    model_config_sha256=hashlib.sha256((Path(cfg['model'])/'config.json').read_bytes()).hexdigest(),
                    image_processor=processor.image_processor.to_dict(),
                    packages={name:version(name) for name in ('torch','transformers','accelerate','flash-attn')},
                    allocator=os.environ['PYTORCH_ALLOC_CONF'],
                    torch=torch.__version__, precision='FP32 trainable/Adam; BF16 autocast',
                    max_episodes=args.max_episodes)
    step = 0
    if args.resume:
        state = load_checkpoint(args.resume, policy, optimizer, scheduler, stream, expected_manifest=manifest)
        step = state['step']; trainer.load_state_dict(state['cursors'])
        expected_step = (stream.epoch - 1)*updates_per_epoch + math.ceil(stream.total_labels/cfg['global_batch_size'])
        if stream.epoch > epochs or step != expected_step:
            raise ValueError('Checkpoint epoch/update accounting mismatch')
        for completed in output.glob('epoch-*/manifest.json'):
            if json.loads(completed.read_text())['step'] > step:
                raise ValueError('A later epoch checkpoint exists; resume the latest checkpoint to avoid stale exports')
    if rank == 0:
        (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(dict(event='ready', world_size=world, states=total_labels, full_states=full_labels,
                              updates=total_updates, trainable_parameters=sum(p.numel() for p in parameters),
                              microbatch=microbatch, execution=execution)), flush=True)
    timings = []
    try:
        while True:
            # Handle completion BEFORE a smoke update limit, including resume
            # from a periodic checkpoint exactly at the epoch boundary.
            if stream.exhausted:
                if trainer.state_dict()['episode'] is not None:
                    raise RuntimeError('Live episode carry at epoch boundary')
                if not args.skip_final_save:
                    epoch_dir = output/f'epoch-{stream.epoch}'
                    if not (epoch_dir/'manifest.json').exists():
                        save_checkpoint(epoch_dir, policy, optimizer, scheduler, stream,
                                        trainer.state_dict(), manifest, step)
                    if rank == 0:
                        marker = epoch_dir/('SMOKE_EPOCH_COMPLETE' if args.max_episodes else 'EPOCH_COMPLETE')
                        if not marker.exists():
                            policy.export(epoch_dir/'export', processor, cfg['recent'])
                            (epoch_dir/'export'/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
                            temporary = marker.with_suffix('.tmp')
                            temporary.write_text(json.dumps(dict(epoch=stream.epoch, step=step))+'\n')
                            temporary.replace(marker)
                        prune_steps(output, stream.epoch, keep=0)
                    if world > 1:
                        dist.barrier()
                if stream.epoch == epochs:
                    break
                stream.advance_epoch()
            if args.max_updates and step >= args.max_updates:
                break
            schedule = stream.plan_update()
            if schedule is None:
                break
            if world > 1:
                dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            trainer.prefetch(schedule.local_segments)
            local_loss = 0.
            scale = world / schedule.global_labels if schedule.global_labels else 0.
            for segment in schedule.local_segments:
                result = trainer.segment(segment, scale)
                local_loss += result['loss_sum']
            if not schedule.global_labels:
                continue
            synchronize_gradients(parameters)
            if not finite_gradients(parameters):
                raise FloatingPointError(f'Nonfinite gradients at update {step + 1}; optimizer not stepped')
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'], foreach=True, error_if_nonfinite=True)
            memory_norm = None
            if policy.memory_enabled:
                memory_norm = float(policy.writer.blocks[0].self_attention.in_proj_weight.grad.norm())
            optimizer.step(); scheduler.step(); step += 1
            if step == 1:
                local_optimizer = getattr(optimizer, 'optim', optimizer)
                adam_tensors = [v for state in local_optimizer.state.values()
                                for name, v in state.items() if name in ('exp_avg', 'exp_avg_sq')]
                assert all(v.dtype == torch.float32 for v in adam_tensors)
                storage = dict(rank=rank, trainable_parameter_bytes=sum(p.numel()*p.element_size() for p in parameters),
                               gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in parameters if p.grad is not None),
                               local_adam_bytes=sum(v.numel()*v.element_size() for v in adam_tensors),
                               frozen_bytes=sum(p.numel()*p.element_size() for p in policy.parameters() if not p.requires_grad))
                (output/f'storage_rank{rank}.json').write_text(json.dumps(storage, indent=2)+'\n')
            loss = torch.tensor(local_loss, device=local_rank, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(loss)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            stats = torch.tensor([elapsed, torch.cuda.max_memory_allocated()/2**30,
                                  torch.cuda.max_memory_reserved()/2**30], device=local_rank)
            if world > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            elapsed, peak, reserved = stats.tolist()
            if schedule.global_labels==cfg['global_batch_size']:
                timings.append(elapsed)
            row = dict(step=step, epoch=stream.epoch, loss=float(loss)/schedule.global_labels,
                       labeled_states=schedule.global_labels, observed_states=schedule.global_observations,
                       consumed_labels=(stream.epoch-1)*total_labels+stream.total_labels, seconds=elapsed,
                       states_per_second=schedule.global_labels/elapsed,
                       grad_norm=float(grad_norm), writer_block0_grad_norm=memory_norm,
                       peak_allocated_gib=peak, peak_reserved_gib=reserved,
                       lr=scheduler.get_last_lr())
            if rank == 0:
                print(json.dumps(row), flush=True)
                with (output/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(output/f'step-epoch-{stream.epoch}-{step}', policy, optimizer, scheduler, stream,
                                trainer.state_dict(), manifest, step)
                if rank == 0:
                    prune_steps(output, stream.epoch, keep=5)
                if world > 1:
                    dist.barrier()
        if not args.skip_final_save:
            complete = stream.epoch == epochs and stream.exhausted
            final_path = output/f'step-epoch-{stream.epoch}-{step}'
            if not complete and not (final_path/'manifest.json').exists():
                save_checkpoint(final_path, policy, optimizer, scheduler, stream,
                                trainer.state_dict(), manifest, step)
            if rank == 0:
                prune_steps(output, stream.epoch, keep=5)
                if complete:
                    # Preserve the existing final-export uploader interface.
                    if not (output/'export').exists():
                        (output/'export').symlink_to(f'epoch-{epochs}/export', target_is_directory=True)
                    marker = 'TRAINING_COMPLETE' if not args.max_episodes else 'SMOKE_COMPLETE'
                    (output/marker).write_text(f'{step} updates; {epochs} complete epochs\n')
        if rank == 0:
            # A resumed single update is a cold-start measurement, not a
            # defensible estimate of full-epoch throughput. Exclude short tails.
            steady = timings[2:] if len(timings)>=5 else []
            report = dict(completed_updates=step, epoch=stream.epoch, configured_epochs=epochs,
                          consumed_labels=(stream.epoch-1)*total_labels+stream.total_labels,
                          full_epoch_labels=full_labels, full_epoch_updates=math.ceil(full_labels/cfg['global_batch_size']),
                          steady_mean_seconds=float(np.mean(steady)) if steady else None,
                          estimated_epoch_hours=float(np.mean(steady))*math.ceil(full_labels/cfg['global_batch_size'])/3600 if steady else None,
                          note='Estimate needs at least five full updates and excludes first two, partial tails, startup, checkpoint/export and contention; smoke is not full-epoch convergence.')
            (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report), flush=True)
    finally:
        loader.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
