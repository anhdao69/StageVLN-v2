"""Bounded CPU overfit diagnostic on one complete, real canonical R2R episode.

Random tiny Qwen weights, reduced native image resolution, and toy learning rates
are deliberate diagnostic settings; this is neither production training nor a
navigation SR/SPL evaluation. Pass criterion is fixed before running: at least
eight whole-episode passes, loss <= 50% of initial, finite gradients and nonzero
initial navigation gradients in every recurrent writer block.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from check_adapter import BASE, tiny_model
from qwen_vl.data.episode_manifest import load_manifest
from qwen_vl.data.episode_stream import EpisodeScheduler
from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.distributed_grad import finite_gradients
from qwen_vl.train.episode_trainer import EpisodeTrainer
from qwen_vl.train.optimizer import build_optimizer


MANIFEST = Path('/groups/yshang/an221229/data/StageVLN-v2/episodes.jsonl')
DATA_ROOT = Path('/groups/yshang/an221229/data/JanusVLN_data')


def shortest_episode():
    """Validate source fingerprint, then validate the selected unchanged row."""
    expected = json.loads(MANIFEST.with_name('data_audit.json').read_text())['files'][MANIFEST.name]
    digest = hashlib.sha256()
    shortest = None
    episodes = observations = labeled = 0
    with MANIFEST.open('rb') as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            episodes += 1
            observations += len(row['frames'])
            labeled += sum(action is not None for action in row['actions'])
            if shortest is None or (len(row['frames']), row['episode']) < (len(shortest['frames']), shortest['episode']):
                shortest = row
    if digest.hexdigest() != expected:
        raise ValueError('Canonical source manifest SHA256 mismatch')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl') as selected:
        selected.write(json.dumps(shortest) + '\n')
        selected.flush()
        episode = load_manifest(selected.name)[0]
    return episode, dict(path=str(MANIFEST), sha256=digest.hexdigest(),
                         episodes=episodes, observations=observations, labeled_states=labeled)


def experiment(report, max_seconds, max_epochs):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    episode, manifest = shortest_episode()
    labels = sum(frame.action is not None for frame in episode.frames)
    report.update(source_manifest=manifest, episode=dict(
        identity=episode.episode, instruction=episode.instruction,
        instruction_sha256=episode.instruction_sha256,
        observations=len(episode.frames), labeled_states=labels,
        actions=[frame.action for frame in episode.frames],
        action_counts=dict(Counter(frame.action for frame in episode.frames)),
        frame_paths=[frame.image_path for frame in episode.frames],
        selection='Shortest complete episode; ties resolved by lexical episode ID'))
    processor = AutoProcessor.from_pretrained(BASE)
    processor.image_processor.size.update(shortest_edge=1024, longest_edge=4096)
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 4096
    memory = dict(slots=4, width=16, layers=3, heads=2, ffn_width=32)
    policy = NavigationPolicy(tiny_model(), processor.tokenizer, memory_enabled=True,
                              memory_config=memory).train()
    optimizer = build_optimizer(policy, lr_language=.003, lr_merger=.003,
                                lr_memory=.003, weight_decay=0.)
    loader = FrameLoader(DATA_ROOT, processor.image_processor, workers=1, cache_size=32)
    trainer = EpisodeTrainer(policy, (episode,), loader, recent=4, execution='bridge',
                             reader_microbatch=2, bf16=False)
    report['configuration'] = dict(
        device='cpu', torch_threads=1, torch_interop_threads=1, loader_workers=1,
        seed=42, initialization='Random tiny Qwen3.5; no pretrained parameter values',
        tokenizer_and_config_source=BASE, dtype='float32', memory=memory,
        K=8, recent=4, execution='bridge', reader_microbatch=2,
        optimizer='AdamW', lr_language=.003, lr_merger=.003, lr_memory=.003,
        weight_decay=0., clip_grad_norm=1., optimizer_steps_per_complete_episode=1,
        min_pixels=1024, max_pixels=4096,
        production_pixel_bounds=[12544, 451584], reduced_pixels_diagnostic_only=True,
        max_seconds=max_seconds, max_epochs=max_epochs,
        trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad))
    writes = [0]
    hook = policy.writer.register_forward_hook(lambda *_: writes.__setitem__(0, writes[0] + 1))
    report['passes'] = []
    report['pass_criterion'] = 'At least 8 complete passes; final loss <= 0.50 * initial loss; finite gradients; nonzero initial writer-block gradients; exact complete chronology and writer counts'
    try:
        loader.prefetch(episode.frames)
        processed_grids = [loader.get(frame)[1].tolist() for frame in episode.frames]
        patch_size = processor.image_processor.patch_size
        with Image.open(DATA_ROOT / episode.frames[0].image_path) as first_image:
            original_size = list(first_image.size)
        report['image_budget'] = dict(
            first_original_wh=original_size, image_grid_thw=processed_grids,
            processed_pixels_per_frame=[grid[1] * grid[2] * patch_size ** 2 for grid in processed_grids],
            native_reader_tokens_per_frame=[int(np.prod(grid)) // 4 for grid in processed_grids],
            raw_images='All original episode RGB files; only native processor resizing reduced')
        initial_block_norms = None
        finite = True
        for epoch in range(max_epochs):
            if report['passes'] and time.monotonic() - report['_started'] >= max_seconds:
                report['stop_reason'] = 'Time budget reached after a completed episode'
                break
            epoch_start = time.monotonic()
            stream = EpisodeScheduler((episode,), seed=42, rank=0, world_size=1,
                                      K=8, target_budget=64, shuffle=False)
            schedule = stream.plan_update()
            assert schedule.global_labels == labels
            assert schedule.global_observations == len(episode.frames)
            assert stream.plan_update() is None
            segments = schedule.local_segments
            assert [t for segment in segments for t in segment.steps] == list(range(len(episode.frames)))
            assert sum(segment.first for segment in segments) == 1
            assert sum(segment.last for segment in segments) == 1
            optimizer.zero_grad(set_to_none=True)
            trainer.prefetch(segments)
            total_loss = 0.
            consumed_labels = consumed_observations = 0
            for segment in segments:
                metrics = trainer.segment(segment, 1. / labels)
                total_loss += metrics['loss_sum']
                consumed_labels += metrics['states']
                consumed_observations += metrics['observations']
                if not segment.last:
                    assert trainer.carry['episode'] == episode.episode
                    assert trainer.carry['memory'].grad_fn is None
                    assert trainer.carry['memory'].dtype == torch.float32
            assert consumed_labels == labels and consumed_observations == len(episode.frames)
            assert trainer.carry['episode'] is None and trainer.carry['memory'] is None
            assert writes[0] == (epoch + 1) * len(episode.frames)
            finite = finite and finite_gradients(policy.parameters())
            writer_norm = float(torch.linalg.vector_norm(torch.stack([
                p.grad.float().norm() for p in policy.writer.parameters() if p.grad is not None])))
            block_norms = [float(torch.linalg.vector_norm(torch.stack([
                p.grad.float().norm() for p in block.parameters() if p.grad is not None])))
                for block in policy.writer.blocks]
            if initial_block_norms is None:
                initial_block_norms = block_norms
            loss = total_loss / labels
            assert np.isfinite(loss) and finite
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.))
            assert np.isfinite(grad_norm)
            optimizer.step()
            record = dict(epoch=epoch + 1, mean_action_state_loss=loss,
                          labels=consumed_labels, observations=consumed_observations,
                          segment_lengths=[len(s.steps) for s in segments],
                          writer_gradient_norm=writer_norm,
                          writer_block_gradient_norms=block_norms,
                          total_gradient_norm_before_clip=grad_norm,
                          seconds=time.monotonic() - epoch_start)
            report['passes'].append(record)
            print(f"pass={epoch + 1} loss={loss:.6f} writer_grad={writer_norm:.6g} seconds={record['seconds']:.2f}",
                  file=sys.stderr, flush=True)
            if epoch + 1 >= 8 and loss <= .5 * report['passes'][0]['mean_action_state_loss']:
                report['stop_reason'] = 'Predeclared loss-reduction criterion achieved'
                break
        else:
            report['stop_reason'] = 'Maximum complete-episode passes reached'
        losses = [record['mean_action_state_loss'] for record in report['passes']]
        report.update(initial_loss=losses[0], final_loss=losses[-1],
                      final_to_initial_ratio=losses[-1] / losses[0],
                      complete_episode_passes=len(losses), writer_calls=writes[0],
                      consumed_observations=writes[0], consumed_labeled_states=labels * len(losses),
                      all_gradients_finite=finite, initial_writer_block_gradient_norms=initial_block_norms,
                      loss_measurement='Before each optimizer step; fixed weights across the entire complete episode',
                      passed=bool(len(losses) >= 8 and losses[-1] <= .5 * losses[0] and finite and
                                  all(norm > 0 and np.isfinite(norm) for norm in initial_block_norms)),
                      limitations=['Random tiny model and toy hyperparameters',
                                   'Reduced native image resolution',
                                   'One episode dominated by MOVE_FORWARD; not evidence of memory benefit',
                                   'No simulator, navigation success rate, SPL, or production-model fit claim'])
    finally:
        hook.remove()
        loader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seconds', type=float, default=240.)
    parser.add_argument('--max-epochs', type=int, default=40)
    args = parser.parse_args()
    report = {'diagnostic': 'Complete real canonical R2R episode overfit',
              '_started': time.monotonic(), 'passed': False}
    try:
        experiment(report, args.max_seconds, args.max_epochs)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc(file=sys.stderr)
    finally:
        report['wall_seconds'] = time.monotonic() - report.pop('_started')
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
