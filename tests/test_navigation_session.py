"""Actor writes once per observation and keeps native generation semantics."""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from transformers import AutoProcessor

from qwen_vl.contracts import EpisodeRecord, FrameKey, FrameRecord
from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.data.prompting import build_state
from qwen_vl.eval.session import PolicySession, decode_action, greedy_generate
from qwen_vl.models.navigation_policy import NavigationPolicy

_spec = importlib.util.spec_from_file_location('adapter_fixture', Path(__file__).resolve().parents[1] / 'scripts/recurrent/check_adapter.py')
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


@pytest.fixture(scope='module')
def processor():
    return AutoProcessor.from_pretrained(_fixture.BASE, min_pixels=1024, max_pixels=4096)


@pytest.fixture
def policy(processor):
    torch.set_num_threads(1)
    torch.manual_seed(14)
    return NavigationPolicy(_fixture.tiny_model(), processor.tokenizer, memory_enabled=True,
                            memory_config=dict(slots=4, width=32, layers=2, heads=4, ffn_width=64)).eval()


@contextmanager
def forced_action(policy, action='STOP'):
    """Keep real reader/cache execution but make random tiny weights emit an action."""
    tokens = policy.tokenizer.encode(action, add_special_tokens=False) + [policy.tokenizer.eos_token_id]
    count = [0]
    def logits(_module, _inputs, output):
        result = torch.full_like(output, -10000.)
        result[..., tokens[count[0] % len(tokens)]] = 0
        count[0] += 1
        return result
    handle = policy.backbone.lm_head.register_forward_hook(logits)
    try:
        yield count
    finally:
        handle.remove()


def test_lifecycle_prefix_causal_recent_duplicates_and_reset(policy, processor, tmp_path):
    loader = FrameLoader(tmp_path, processor.image_processor)
    session = PolicySession(policy, loader, recent=4, bf16=False)
    writes = []
    handle = policy.writer.register_forward_hook(lambda _m, _i, output: writes.append(output.state.clone()))
    try:
        with forced_action(policy) as generations:
            session.reset('episode-a', 'Turn left at the door.')
            image = Image.fromarray(np.full((32, 48, 3), 70, dtype=np.uint8))
            for step in range(6):
                rgb = np.asarray(image) if step == 2 else image
                assert session.observe(step, rgb) == 'STOP'
                assert [key.step for key in session.last_state.frame_keys] == list(range(max(0, step - 4), step + 1))
                assert session.last_state.labels.eq(-100).all()
            memory = session.memory.clone()
            assert session.observe(3, image) == 'STOP'
            assert len(writes) == 6 and generations[0] > len(writes)
            torch.testing.assert_close(session.memory, memory, rtol=0, atol=0)
            assert session.memory.dtype == torch.float32 and session.memory.grad_fn is None
            with pytest.raises(ValueError, match='instruction'):
                session.reset('episode-a', 'A conflicting instruction.')
            with pytest.raises(ValueError, match='conflict'):
                session.observe(3, Image.new('RGB', image.size, color='blue'))
            with pytest.raises(ValueError, match='chronolog'):
                session.observe(7, image)
            # Teacher forcing must have exactly the same prefix, despite targets.
            frames = tuple(FrameRecord(key, '', 'TURN_LEFT', '') for key in session.last_state.frame_keys)
            ep = EpisodeRecord('episode-a', 'Turn left at the door.', '', frames)
            grids = [span.grid_thw for span in session.last_state.image_spans]
            teacher = build_state(ep, frames, policy.tokenizer, grids, policy.memory_slots)
            torch.testing.assert_close(teacher.input_ids[:teacher.action_start], session.last_state.input_ids)
            session.reset('episode-b', 'Turn left at the door.')
            assert session.memory is None and session.recent_keys == ()
            assert session.observe(0, image) == 'STOP'
            torch.testing.assert_close(session.memory, writes[0], rtol=0, atol=0)
    finally:
        handle.remove()
        loader.close()


def test_paths_ignore_labels_and_interleaved_sessions(policy, processor, tmp_path):
    image = Image.new('RGB', (32, 48), color='red')
    image.save(tmp_path / 'frame.png')
    loader = FrameLoader(tmp_path, processor.image_processor)
    a, b = PolicySession(policy, loader), PolicySession(policy, loader)
    try:
        with forced_action(policy):
            a.reset('a', 'Go forward.')
            b.reset('b', 'Go forward.')
            fa = FrameRecord(FrameKey('r2r', 'a', 0), 'frame.png', 'TURN_LEFT', 'source')
            fb = replace(fa, key=FrameKey('r2r', 'b', 0), action='TURN_RIGHT')
            assert a.observe(0, fa) == b.observe(0, fb) == 'STOP'
            torch.testing.assert_close(a.memory, b.memory, rtol=0, atol=0)
            assert a.observe(0, replace(fa, action='STOP')) == 'STOP'
            saved_b = b.memory.clone()
            assert a.observe(1, image) == 'STOP'
            torch.testing.assert_close(b.memory, saved_b, rtol=0, atol=0)
            assert len(a.last_state.image_spans) == 1
            image.paste('blue', (0, 0, 32, 48))
            image.save(tmp_path / 'frame.png')
            with pytest.raises(ValueError, match='conflict'):
                b.observe(0, fb)
    finally:
        loader.close()


@torch.no_grad()
def test_manual_generation_matches_native_positions_logits_and_cache_isolation(policy):
    policy.memory_enabled = False
    model = policy.backbone
    for count in (1, 2, 5, 9):
        frames = tuple(FrameRecord(FrameKey('r2r', str(count), i), '', None, '') for i in range(count))
        ep = EpisodeRecord(str(count), 'Find the door.', '', frames)
        grids = torch.tensor([(1, 4, 6), (1, 6, 4)] * (count // 2) + ([(1, 4, 6)] if count % 2 else []))
        pixels = torch.randn(int(grids.prod(-1).sum()), 3 * 2 * 16 * 16)
        features = {f.key: f for f in policy.visual.encode(pixels, grids, [f.key for f in frames])}
        state = build_state(ep, frames, policy.tokenizer, grids.tolist(), include_target=False)
        inputs, _, native = policy.reader.prepare([state], features, supervised=False)
        native_output = model.generate(**native, pixel_values=pixels, use_cache=True, max_new_tokens=4,
                                       do_sample=False, eos_token_id=None, pad_token_id=policy.tokenizer.pad_token_id,
                                       return_dict_in_generate=True, output_logits=True)
        calls = []
        def record(_module, _args, kwargs, output):
            calls.append((kwargs['position_ids'].clone(), kwargs['cache_position'].clone(),
                          kwargs.get('past_key_values'), output.past_key_values, output.logits[:, -1].clone()))
        handle = model.register_forward_hook(record, with_kwargs=True)
        try:
            model.model.rope_deltas = torch.tensor([[999]])  # Must not affect explicit positions.
            actual = greedy_generate(model, inputs, max_new_tokens=4, eos_token_ids=())
            repeated = greedy_generate(model, inputs, max_new_tokens=4, eos_token_ids=())
        finally:
            handle.remove()
        torch.testing.assert_close(actual, native_output.sequences[:, len(state.input_ids):], rtol=0, atol=0)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        assert calls[0][2] is None and calls[4][2] is None
        assert calls[0][3] is not calls[4][3]
        assert model.model.rope_deltas is None
        length = len(state.input_ids)
        next_position = int(inputs['position_ids'].max()) + 1
        for j, (positions, cache_position, _, _, logits) in enumerate(calls[:4]):
            torch.testing.assert_close(logits, native_output.logits[j], atol=1e-5, rtol=1e-4)
            if j:
                assert cache_position.tolist() == [length + j - 1]
                assert positions[-3:, 0, 0].tolist() == [next_position + j - 1] * 3


def test_invalid_action_is_rejected_without_duplicate_write(policy, processor, tmp_path):
    loader = FrameLoader(tmp_path, processor.image_processor)
    session = PolicySession(policy, loader)
    count = []
    handle = policy.writer.register_forward_hook(lambda *_: count.append(1))
    try:
        session.reset('episode', 'Go forward.')
        image = Image.new('RGB', (32, 48))
        with forced_action(policy, 'LEFT'):
            with pytest.raises(ValueError, match='action'):
                session.observe(0, image)
            with pytest.raises(ValueError, match='action'):
                session.observe(0, image)
        assert len(count) == 1
        for text in ('left', 'STOP now', 'TURN_LEFT TURN_RIGHT', ''):
            with pytest.raises(ValueError):
                decode_action(text)
        assert decode_action(' TURN_LEFT\n') == 'TURN_LEFT'
    finally:
        handle.remove()
        loader.close()


def test_export_reload_preserves_memory_prefix_and_greedy_tokens(policy, processor, tmp_path):
    path = tmp_path / 'export'
    # CPU fixture owns a plain module; Accelerate's optional DeepSpeed import
    # otherwise probes unavailable CUDA build tooling while merely saving it.
    with patch('accelerate.utils.other.is_deepspeed_available', return_value=False):
        policy.export(path, processor, recent=4)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
    with patch.multiple(qm, causal_conv1d_fn=None, causal_conv1d_update=None,
                        chunk_gated_delta_rule=None, fused_recurrent_gated_delta_rule=None, FusedRMSNormGated=None):
        restored, restored_processor, recent = NavigationPolicy.from_export(path, dtype=torch.float32)
    assert recent == 4
    loaders = [FrameLoader(tmp_path, p.image_processor) for p in (processor, restored_processor)]
    sessions = [PolicySession(p, loader, recent=recent) for p, loader in zip((policy, restored), loaders)]
    image = Image.new('RGB', (32, 48), color='green')
    try:
        for session in sessions:
            session.reset('episode', 'Turn right.')
            with forced_action(session.policy):
                session.observe(0, image)
        torch.testing.assert_close(sessions[0].memory, sessions[1].memory, rtol=0, atol=0)
        torch.testing.assert_close(sessions[0].last_state.input_ids, sessions[1].last_state.input_ids)
        with torch.inference_mode():
            unforced_tokens = []
            for session in sessions:
                adapted = session.policy.memory_adapter(session.memory)[0]
                unforced_tokens.append(session.generate_greedy(session.last_state, session._features, adapted))
            torch.testing.assert_close(unforced_tokens[0], unforced_tokens[1], rtol=0, atol=0)
        for name, param in restored.named_parameters():
            torch.testing.assert_close(param, policy.state_dict()[name], rtol=0, atol=0)
        assert len(list(restored.parameters())) == len({id(p) for p in restored.parameters()})
    finally:
        for loader in loaders:
            loader.close()
