# Inference actor/session implementation

Owned changes only: `src/qwen_vl/eval/__init__.py`, `src/qwen_vl/eval/session.py`, and `tests/test_navigation_session.py`.

## Integration

```python
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.eval import PolicySession

policy, processor, recent = NavigationPolicy.from_export(export_dir, device='cuda')
loader = FrameLoader(image_root, processor.image_processor)
session = PolicySession(policy, loader, recent=recent, max_new_tokens=16)
session.reset(episode_id, instruction)
action = session.observe(0, image)  # PIL RGB, uint8 HWC array, path, or FrameRecord
# Repeat for contiguous step IDs. Caller owns loader.close().
```

- `reset(episode_id, instruction)` clears recurrent state and recent features, recomputes fixed-weight instruction features, and records a new episode identity. Reusing the same episode identity with conflicting instruction raises; explicit reset with unchanged instruction is allowed.
- `observe(step_id, rgb)` writes current visual information exactly once, renders the label-free memory/current or memory/recent4/current prefix, and greedily generates one primitive action. R4 reads four previous observations before appending current. R0 retains no recent visual features.
- Identical repeated steps, including older steps, return the previous action. RGB hashes are based on decoded pixels and dimensions, so changed bytes at an existing path are detected. Supplied FrameRecord actions are discarded. New steps must be contiguous, starting at zero.
- `decode_action(text)` accepts exactly MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, or STOP after stripping surrounding whitespace; the session removes tokenizer special tokens first. Invalid outputs raise, with no fallback. The invalid decision is remembered so retries do not write again. Stored errors have no traceback, preventing retention of observation tensors.
- `memory` is detached FP32. `recent_keys`, `last_state`, and `last_token_ids` expose identities/prefix/tokens for diagnostics. The policy must remain fixed while a session is active. Sessions can interleave calls on one shared policy sequentially; concurrent calls are unsupported.
- `generate_greedy(state, features, memory=None)` is available on the session for raw token generation without a write. `greedy_generate(backbone, prepared_inputs, max_new_tokens=16, eos_token_ids=None)` returns only generated token IDs `[1, n]`, including EOS if produced.

## Native generation contract

Inspected installed Transformers 5.3 Qwen3.5 forward/generation implementation. Reader.prepare supplies explicit three-axis RoPE positions. The actor prepends the native sequential text axis, yielding `[4, 1, length]`, and supplies `cache_position=arange(length)` for inputs_embeds prefill with no existing cache. Continuation token j uses new token IDs only, text/cache position L+j, and three RoPE coordinates max(prefix_RoPE)+1+j. It does not send pixels or call the writer during generated-token continuation. The cache exists only in the local generation function; native rope_deltas are cleared on entry and exit.

## Validation

Tests exercise the real tiny Qwen3.5 model and actual tokenizer/image processor. Tiny actor tests replace only the final logits with a known action to make random weights emit valid actions while retaining real reader/cache/writer execution. Separate unmodified-logit tests compare manual versus native generation token IDs and every generated-token logit for 1, 2, 5, and 9 mixed-aspect-ratio images, using atol=1e-5/rtol=1e-4. They also verify cache separation and continuation positions.

Lifecycle tests cover six chronological observations, causal R4 history, R0, duplicate old steps, conflicting RGB/instruction, skipped steps, reset, no writes from generated tokens, detached FP32 state, teacher-forcing/inference prefix equality, ignored target labels, independent interleaved sessions, and invalid-action retries. Actual export/reload checks exact memory, prefix, parameters, unique parameter identities, and unforced greedy token IDs.

CPU-only test accommodations: tiny model construction/reload uses official PyTorch recurrent implementations by disabling optional CUDA FLA functions at initialization. Export uses a test-only `accelerate.utils.other.is_deepspeed_available=False` patch: otherwise Accelerate's plain-module unwrapping imports installed DeepSpeed and fails probing absent CUDA_HOME on the login node. Production policy/export/adapter code was not changed. TorchScript deprecation warnings originate in installed dependencies.

No actual-Qwen GPU continuation, simulator adapter, closed-loop SR/SPL, or deployment was run in this scoped task.

Final command:

```sh
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 /home/an221229/code/SpatialForcing-VLN/.venv/bin/python -m pytest tests/test_navigation_session.py tests/test_episode_stream.py tests/test_distributed_grad.py tests/test_temporal_checkpoint.py -q
```

Result: **15 passed, 14 dependency deprecation warnings in 34.69s** (5 actor tests plus 10 temporal support regression tests).
