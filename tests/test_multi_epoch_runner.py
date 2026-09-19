"""Exercise the real orchestration with tiny CPU components and real saves."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from qwen_vl.train import run_episode as runner
from test_episode_stream import episodes


def test_runner_five_epoch_saves_and_interrupted_resume(monkeypatch, tmp_path):
    data = episodes([['STOP']*3])
    source = tmp_path/'episodes.jsonl'; source.write_text('fixture')
    model_dir = tmp_path/'base'; model_dir.mkdir(); (model_dir/'config.json').write_text('{}')
    cfg = json.loads((Path(__file__).parents[1]/'configs/datasets/v2_mem64_r0_5epochs.json').read_text())
    cfg.update(model=str(model_dir), episode_manifest=str(source), global_batch_size=2,
               memory_enabled=False, optimizer_sharding=False, attention='sdpa')
    config = tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    monkeypatch.setattr(runner, 'load_manifest', lambda *args: data)
    class ImageProcessor:
        size = {}
        def to_dict(self): return {}
    processor = SimpleNamespace(image_processor=ImageProcessor(), tokenizer=SimpleNamespace(chat_template='fixture'))
    monkeypatch.setattr(runner.AutoProcessor, 'from_pretrained', lambda *a, **k: processor)
    class Backbone:
        def __init__(self):
            visual = torch.nn.Linear(1,1)
            visual.merger = torch.nn.Linear(1,1)
            self.model = SimpleNamespace(visual=visual)
    monkeypatch.setattr(runner.Qwen3_5ForConditionalGeneration, 'from_pretrained', lambda *a, **k: Backbone())
    class Policy(torch.nn.Module):
        memory_enabled = False
        def __init__(self, backbone, tokenizer, enabled):
            super().__init__(); self.layer = torch.nn.Linear(2,1); self.tokenizer = tokenizer
        def to(self, **kwargs): return self
        def export(self, path, processor, recent):
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path/'weights.pt')
    monkeypatch.setattr(runner, 'NavigationPolicy', Policy)
    monkeypatch.setattr(runner, 'build_optimizer', lambda policy, **kwargs: torch.optim.AdamW(policy.parameters(), lr=.01))
    monkeypatch.setattr(runner, 'FrameLoader', lambda *a, **k: SimpleNamespace(close=lambda: None))
    class Trainer:
        def __init__(self, policy, *args): self.policy = policy; self.episode = None
        def prefetch(self, segments): pass
        def segment(self, segment, scale):
            self.episode = None if segment.last else '0'
            loss = self.policy.layer(torch.randn(len(segment.steps),2)).square().sum()
            (loss*scale).backward()
            return {'loss_sum':float(loss.detach())}
        def state_dict(self): return {'episode':self.episode}
        def load_state_dict(self, state): self.episode = state['episode']
    monkeypatch.setattr(runner, 'EpisodeTrainer', Trainer)
    for name in ['set_device','synchronize','reset_peak_memory_stats']:
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    for name in ['max_memory_allocated','max_memory_reserved']:
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    original_tensor = torch.tensor
    def cpu_tensor(*args, **kwargs):
        kwargs.pop('device', None)
        return original_tensor(*args, **kwargs)
    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    monkeypatch.setenv('WORLD_SIZE','1'); monkeypatch.setenv('RANK','0'); monkeypatch.setenv('LOCAL_RANK','0')
    def run(output, limit=0, resume=None):
        args = SimpleNamespace(config=str(config), output=str(output), max_updates=limit,
            max_episodes=0, checkpoint_every=1, skip_final_save=False,
            no_gradient_checkpointing=False, execution=None, reader_microbatch=None, resume=resume)
        monkeypatch.setattr(runner, 'arguments', lambda: args)
        runner.main()
    uninterrupted = tmp_path/'uninterrupted'
    run(uninterrupted)
    resumed = tmp_path/'resumed'
    run(resumed, limit=3)
    assert not (resumed/'TRAINING_COMPLETE').exists()
    run(resumed, resume=str(resumed/'step-epoch-2-3'))
    for output in [uninterrupted, resumed]:
        assert (output/'TRAINING_COMPLETE').exists()
        assert not list(output.glob('step-epoch-*'))
        for epoch in range(1,6):
            completion = json.loads((output/f'epoch-{epoch}/manifest.json').read_text())
            assert (completion['epoch'],completion['step']) == (epoch,epoch*2)
            assert (output/f'epoch-{epoch}/EPOCH_COMPLETE').exists()
    expected = torch.load(uninterrupted/'epoch-5/model.pt', weights_only=True)
    actual = torch.load(resumed/'epoch-5/model.pt', weights_only=True)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
