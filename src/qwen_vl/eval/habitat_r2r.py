"""Closed-loop Habitat R2R-VLNCE evaluation for exported recurrent policies.

Habitat is imported lazily because its audited 0.2.4 runtime uses Python 3.10,
while model training uses the project's Python 3.12 uv environment. PolicySession
writes each current observation once, then reads post-observation memory. The
export selects either the v2 current-only reader or the v3 Recent4 reader.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from qwen_vl.data.frame_loader import FrameLoader
from qwen_vl.data.data_qwen import QWEN3_5_NON_THINKING_CHAT_TEMPLATE
from qwen_vl.data.history import history_indices
from qwen_vl.eval.session import ACTIONS, InvalidActionError, PolicySession
from qwen_vl.eval.uniform_session import UniformHistorySession
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.train_qwen import _install_qwen35_flash_attention_fix


HABITAT_ACTIONS = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_id(scene_path: str) -> str:
    path = Path(scene_path)
    return path.parent.name if path.suffix else path.name


def _agent_pose(env) -> dict:
    state = env.sim.get_agent_state()
    rotation = state.rotation
    # Habitat-Sim 0.2.4 exposes numpy-quaternion as scalar real + xyz imag.
    return {
        "position_xyz": [float(value) for value in state.position],
        "rotation_xyzw": [float(value) for value in rotation.imag] + [float(rotation.real)],
    }


def _trajectory_name(scene_id: str, episode_id: str) -> str:
    safe = lambda value: "".join(char if char.isalnum() or char in "-_." else "_"
                                  for char in str(value))
    return f"{safe(scene_id)}__{safe(episode_id)}.json"


def _write_atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def validate_export(path: Path) -> dict:
    """Require a complete Memory64 R2R export with its trained reader window."""
    required = [
        path / "backbone" / "config.json",
        path / "backbone" / "processor_config.json",
        path / "backbone" / "tokenizer.json",
        path / "backbone" / "model.safetensors.index.json",
        path / "memory.pt",
        path / "navigation_config.json",
        path / "prompt_protocol.json",
        path / "run_manifest.json",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ValueError("Incomplete policy export:\n  - " + "\n  - ".join(missing))
    navigation = json.loads((path / "navigation_config.json").read_text())
    prompt = json.loads((path / "prompt_protocol.json").read_text())
    run = json.loads((path / "run_manifest.json").read_text())
    expected_memory = {"slots": 64, "width": 512, "layers": 3,
                       "heads": 8, "ffn_width": 2048}
    if not navigation.get("memory_enabled"):
        raise ValueError("R2R evaluation requires an enabled recurrent memory")
    if navigation.get("memory") != expected_memory:
        raise ValueError(f"Unexpected memory architecture: {navigation.get('memory')}")
    if tuple(prompt.get("actions", ())) != ACTIONS:
        raise ValueError("Exported action protocol differs from PolicySession")
    if prompt.get("chat_template") != QWEN3_5_NON_THINKING_CHAT_TEMPLATE:
        raise ValueError("Exported chat template differs from the inference renderer")
    config = run.get("config", {})
    expected_recent = {"v2_mem64_r0": 0, "v3_mem64_r4": 4}
    variant = config.get("variant")
    if variant not in expected_recent:
        raise ValueError(f"Unsupported R2R policy variant: {variant!r}")
    recent = expected_recent[variant]
    if navigation.get("recent") != recent or prompt.get("recent") != recent or config.get("recent") != recent:
        raise ValueError(f"Reader window disagrees with {variant}: expected recent={recent}")
    if config.get("memory_enabled") is not True or run.get("memory_architecture") != expected_memory:
        raise ValueError("Run manifest memory architecture disagrees with export")
    if run.get("max_episodes") != 0:
        raise ValueError("Refusing benchmark evaluation of a smoke-only export")
    return {"navigation": navigation, "prompt": prompt, "run": run}


def validate_v0_export(path: Path, history: str = "uniform4") -> dict:
    """Validate a v0 SFT artifact against its exact trained history rule."""
    if history not in ("uniform4", "uniform8"):
        raise ValueError(f"Unsupported v0 history rule: {history!r}")
    required = ["config.json", "processor_config.json", "tokenizer.json",
                "prompt_protocol.json", "training_metadata.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete v0 export: {missing}")
    if not (path / "model.safetensors").is_file() and not (path / "model.safetensors.index.json").is_file():
        raise ValueError("Incomplete v0 export: model safetensors are missing")
    prompt = json.loads((path / "prompt_protocol.json").read_text())
    training = json.loads((path / "training_metadata.json").read_text())
    if training.get("variant") != f"v0_{history}" or prompt.get("dataset_config", {}).get("history") != history:
        raise ValueError(f"Checkpoint was not trained with {history.title()} history")
    if tuple(prompt.get("actions", ())) != ACTIONS:
        raise ValueError("Exported v0 action protocol differs from the evaluator")
    if prompt.get("chat_template") != QWEN3_5_NON_THINKING_CHAT_TEMPLATE:
        raise ValueError("Exported v0 chat template differs from the inference renderer")
    expected_protocol = {
        "system": "You are a helpful assistant.",
        "assistant_prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "target_suffix": "<|im_end|>\n",
        "template_sha256": hashlib.sha256(QWEN3_5_NON_THINKING_CHAT_TEMPLATE.encode()).hexdigest(),
    }
    if any(prompt.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("Exported v0 prompt protocol differs from the training renderer")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("v0 export is not a Qwen3.5 conditional-generation model")
    return {"prompt": prompt, "training": training}


def validate_v1_export(path: Path) -> dict:
    """Require the memory-free v1 SFT weights and saved Sliding Window 4 prompt."""
    required = ["config.json", "processor_config.json", "tokenizer.json",
                "prompt_protocol.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete v1 export: {missing}")
    if not (path / "model.safetensors").is_file() and not (path / "model.safetensors.index.json").is_file():
        raise ValueError("Incomplete v1 export: model safetensors are missing")
    prompt = json.loads((path / "prompt_protocol.json").read_text())
    if prompt.get("dataset_config", {}).get("history") != "sw4":
        raise ValueError("Checkpoint was not trained with Sliding Window 4 history")
    if tuple(prompt.get("actions", ())) != ACTIONS:
        raise ValueError("Exported v1 action protocol differs from the evaluator")
    expected_protocol = {
        "system": "You are a helpful assistant.",
        "chat_template": QWEN3_5_NON_THINKING_CHAT_TEMPLATE,
        "assistant_prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "target_suffix": "<|im_end|>\n",
        "template_sha256": hashlib.sha256(QWEN3_5_NON_THINKING_CHAT_TEMPLATE.encode()).hexdigest(),
    }
    if any(prompt.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("Exported v1 prompt protocol differs from the training renderer")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("v1 export is not a Qwen3.5 conditional-generation model")
    training_path = path / "training_metadata.json"
    training = json.loads(training_path.read_text()) if training_path.is_file() else None
    if training and training.get("variant") not in (None, "v1_sw4", "v1_r2r_sw4"):
        raise ValueError("v1 training metadata disagrees with Sliding Window 4")
    return {"prompt": prompt, "training": training}


def validate_plain_processor(processor, path: Path) -> None:
    saved = json.loads((path / "processor_config.json").read_text())["image_processor"]
    actual = processor.image_processor.to_dict()
    keys = ("size", "patch_size", "temporal_patch_size", "merge_size",
            "image_mean", "image_std")
    mismatches = {key: (saved.get(key), actual.get(key)) for key in keys
                  if saved.get(key) != actual.get(key)}
    size = actual.get("size", {})
    for key, edge in (("min_pixels", "shortest_edge"), ("max_pixels", "longest_edge")):
        if saved.get(key) != actual.get(key, size.get(edge)):
            mismatches[key] = (saved.get(key), actual.get(key, size.get(edge)))
    if mismatches:
        raise ValueError(f"Plain-policy processor differs from training export: {mismatches}")


def validate_processor(processor, manifest: dict) -> None:
    """Check inference preprocessing against the processor recorded at training."""
    actual = processor.image_processor.to_dict()
    expected = manifest["run"].get("image_processor", {})
    keys = ("patch_size", "temporal_patch_size", "merge_size", "image_mean",
            "image_std")
    mismatches = {key: (expected.get(key), actual.get(key)) for key in keys
                  if expected.get(key) != actual.get(key)}
    expected_size, actual_size = expected.get("size", {}), actual.get("size", {})
    if expected_size != actual_size:
        mismatches["size"] = (expected_size, actual_size)
    # Transformers restores these bounds as size.shortest/longest_edge and may
    # omit the redundant top-level fields from image_processor.to_dict().
    effective_bounds = {
        "min_pixels": actual.get("min_pixels", actual_size.get("shortest_edge")),
        "max_pixels": actual.get("max_pixels", actual_size.get("longest_edge")),
    }
    for key, value in effective_bounds.items():
        if expected.get(key) != value:
            mismatches[key] = (expected.get(key), value)
    if mismatches:
        raise ValueError(f"Exported processor does not match training manifest: {mismatches}")


def _load_completed(path: Path, identity: dict) -> tuple[list[dict], set[tuple[str, str]]]:
    records = []
    completed = set()
    if not path.exists():
        return records, completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("identity") != identity:
                raise ValueError(f"Result identity mismatch at {path}:{line_number}")
            key = (str(row["scene_id"]), str(row["episode_id"]))
            if key in completed:
                raise ValueError(f"Duplicate episode result in {path}: {key}")
            completed.add(key)
            records.append(row)
    return records, completed


def summarize(records: list[dict], identity: dict) -> dict:
    action_counts = Counter()
    latencies = []
    for row in records:
        action_counts.update(row.get("predicted_action_counts", {}))
        latencies.extend(row.get("step_times_ms", ()))
    count = len(records)
    mean = lambda key: (sum(float(row[key]) for row in records) / count) if count else 0.0
    return {
        "identity": identity,
        "episodes": count,
        "success": mean("success"),
        "spl": mean("spl"),
        "oracle_success": mean("oracle_success"),
        "navigation_error": mean("navigation_error"),
        "total_steps": sum(int(row["steps"]) for row in records),
        "invalid_outputs": sum(int(row["invalid_outputs"]) for row in records),
        "forced_max_step_stops": sum(int(row["forced_max_step_stop"]) for row in records),
        "predicted_action_counts": dict(sorted(action_counts.items())),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
        },
    }


def _build_config(args):
    import habitat
    from habitat.config.default import get_config

    config = get_config(str(args.habitat_config))
    with habitat.config.read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.data_path = str(args.data_path)
        config.habitat.dataset.scenes_dir = str(args.scenes_dir)
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
        config.habitat.environment.max_episode_steps = args.max_steps
        if args.scenes:
            config.habitat.dataset.content_scenes = sorted(args.scenes.split(","))
    return config


def _make_env(config, max_episodes: int):
    from habitat import Env
    from habitat.datasets import make_dataset

    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    if not dataset.episodes:
        raise ValueError("The selected R2R split has no episodes")
    if max_episodes:
        dataset.episodes = dataset.episodes[:max_episodes]
    return Env(config=config, dataset=dataset)


@torch.inference_mode()
def evaluate_checkpoint(args, checkpoint: Path) -> dict:
    is_v0 = args.policy in ("v0_uniform4", "v0_uniform8")
    is_v1 = args.policy == "v1_sw4"
    is_plain = is_v0 or is_v1
    v0_history = args.policy.removeprefix("v0_") if is_v0 else None
    metadata = (validate_v0_export(checkpoint, v0_history) if is_v0 else
                validate_v1_export(checkpoint) if is_v1 else validate_export(checkpoint))
    checkpoint_name = checkpoint.parent.name if checkpoint.name == "export" else checkpoint.name
    identity = {
        "checkpoint": str(checkpoint.resolve()),
        "split": args.split,
        "max_steps": args.max_steps,
        "max_episodes": args.max_episodes,
        "seed": args.seed,
        "invalid_action": "STOP",
        "protocol": "R2R-VLNCE-0.2.4",
        "trajectory_format": 1,
    }
    if is_plain:
        weight_index = checkpoint / "model.safetensors.index.json"
        identity.update({
            "variant": args.policy,
            "history": v0_history if is_v0 else "sw4",
            "weights_sha256": _sha256(weight_index if weight_index.is_file() else checkpoint / "model.safetensors"),
            "prompt_protocol_sha256": _sha256(checkpoint / "prompt_protocol.json"),
            "attention": args.attention,
            "max_new_tokens": args.max_new_tokens,
            "trajectory_format": 2,
        })
        training_path = checkpoint / "training_metadata.json"
        if training_path.is_file():
            identity["training_metadata_sha256"] = _sha256(training_path)
    else:
        identity["memory_sha256"] = _sha256(checkpoint / "memory.pt")
    if not is_plain and metadata["navigation"]["recent"] == 4:
        identity.update({
            "backbone_index_sha256": _sha256(checkpoint / "backbone" / "model.safetensors.index.json"),
            "run_manifest_sha256": _sha256(checkpoint / "run_manifest.json"),
            "prompt_protocol_sha256": _sha256(checkpoint / "prompt_protocol.json"),
            "variant": "v3_mem64_r4",
            "recent": 4,
            "attention": args.attention,
            "max_new_tokens": args.max_new_tokens,
            "trajectory_format": 2,
        })
    output_dir = args.output / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    records, completed = _load_completed(result_path, identity)

    config = _build_config(args)
    env = _make_env(config, args.max_episodes)
    try:
        episodes_by_scene = {}
        for episode in env.episodes:
            episodes_by_scene.setdefault(episode.scene_id, []).append(episode)
        expected_keys = {(str(_scene_id(ep.scene_id)), str(ep.episode_id))
                         for values in episodes_by_scene.values() for ep in values}
        extra = completed - expected_keys
        if extra:
            raise ValueError(f"Results contain episodes outside the selected benchmark: {sorted(extra)[:3]}")
        missing_trajectories = [
            trajectory_dir / _trajectory_name(scene_id, episode_id)
            for scene_id, episode_id in completed
            if not (trajectory_dir / _trajectory_name(scene_id, episode_id)).is_file()
        ]
        if missing_trajectories:
            raise ValueError(
                "Completed metrics are missing trajectory files:\n  - "
                + "\n  - ".join(str(path) for path in missing_trajectories[:20])
            )

        if completed != expected_keys:
            _install_qwen35_flash_attention_fix()
            loader = None
            if is_plain:
                from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

                processor = AutoProcessor.from_pretrained(checkpoint)
                validate_plain_processor(processor, checkpoint)
                processor.tokenizer.chat_template = QWEN3_5_NON_THINKING_CHAT_TEMPLATE
                backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
                    checkpoint, dtype=torch.bfloat16,
                    attn_implementation=args.attention,
                ).to(args.device).eval()
                session = UniformHistorySession(backbone, processor,
                                                max_new_tokens=args.max_new_tokens,
                                                history_mode=v0_history if is_v0 else "recent")
                policy = session
                recent = None
            else:
                policy, processor, recent = NavigationPolicy.from_export(
                    checkpoint, device=args.device, dtype=torch.bfloat16,
                    attn_implementation=args.attention,
                )
                validate_processor(processor, metadata)
                if recent != metadata["navigation"]["recent"]:
                    raise ValueError("Loaded reader window disagrees with export metadata")
                loader = FrameLoader(Path.cwd(), processor.image_processor, workers=1, cache_size=1)
                session = PolicySession(policy, loader, recent=recent,
                                        max_new_tokens=args.max_new_tokens, bf16=True)
            try:
                with result_path.open("a", buffering=1) as output:
                    episode_index = len(completed)
                    for scene_path in sorted(episodes_by_scene):
                        scene_id = _scene_id(scene_path)
                        for episode in episodes_by_scene[scene_path]:
                            key = (str(scene_id), str(episode.episode_id))
                            if key in completed:
                                continue
                            instruction = episode.instruction.instruction_text.strip()
                            env.current_episode = episode
                            observations = env.reset()
                            session.reset(f"{scene_id}:{episode.episode_id}", instruction)
                            initial = env.get_metrics()
                            min_distance = float(initial["distance_to_goal"])
                            trajectory = {
                                "identity": identity,
                                "scene_id": str(scene_id),
                                "episode_id": str(episode.episode_id),
                                "instruction": instruction,
                                "start": {
                                    **_agent_pose(env),
                                    "distance_to_goal": min_distance,
                                },
                                "transitions": [],
                            }
                            actions = Counter()
                            invalid_samples = []
                            invalid_outputs = 0
                            forced_stop = False
                            step_times = []
                            started = time.perf_counter()
                            step = 0
                            while not env.episode_over and step < args.max_steps:
                                tick = time.perf_counter()
                                try:
                                    predicted = session.observe(step, observations["rgb"])
                                    raw_output = policy.tokenizer.decode(
                                        session.last_token_ids[0].tolist(), skip_special_tokens=True
                                    )
                                    invalid_output = False
                                except InvalidActionError:
                                    invalid_outputs += 1
                                    raw = policy.tokenizer.decode(
                                        session.last_token_ids[0].tolist(), skip_special_tokens=True
                                    ) if session.last_token_ids is not None else ""
                                    if len(invalid_samples) < 3:
                                        invalid_samples.append(raw)
                                    predicted = "STOP"
                                    raw_output = raw
                                    invalid_output = True
                                reader_frame_steps = [key.step for key in session.last_state.frame_keys]
                                if is_plain or recent == 4:
                                    expected_frame_steps = (
                                        history_indices(step, v0_history if is_v0 else "recent", recent=4)
                                        if is_plain else list(range(max(0, step - recent), step + 1)))
                                    if reader_frame_steps != expected_frame_steps:
                                        raise RuntimeError(
                                            f"Reader window at step {step}: {reader_frame_steps}, "
                                            f"expected {expected_frame_steps}"
                                        )
                                if predicted not in HABITAT_ACTIONS:
                                    raise RuntimeError(f"Actor returned unknown action {predicted!r}")
                                actions[predicted] += 1
                                executed = predicted
                                if step + 1 >= args.max_steps:
                                    executed = "STOP"
                                    forced_stop = predicted != "STOP"
                                observations = env.step(HABITAT_ACTIONS[executed])
                                step_times.append((time.perf_counter() - tick) * 1000)
                                step += 1
                                metrics = env.get_metrics()
                                min_distance = min(min_distance, float(metrics["distance_to_goal"]))
                                transition = {
                                    "step": step - 1,
                                    "predicted_action": predicted,
                                    "executed_action": executed,
                                    "raw_output": raw_output,
                                    "invalid_output": invalid_output,
                                    **_agent_pose(env),
                                    "distance_to_goal": float(metrics["distance_to_goal"]),
                                }
                                if is_plain or recent == 4:
                                    transition["reader_frame_steps"] = reader_frame_steps
                                trajectory["transitions"].append(transition)
                            metrics = env.get_metrics()
                            success_distance = float(
                                config.habitat.task.measurements.success.success_distance
                            )
                            row = {
                                "identity": identity,
                                "scene_id": str(scene_id),
                                "episode_id": str(episode.episode_id),
                                "instruction": instruction,
                                "success": float(metrics["success"]),
                                "spl": float(metrics["spl"]),
                                "oracle_success": float(min_distance < success_distance),
                                "navigation_error": float(metrics["distance_to_goal"]),
                                "minimum_navigation_error": min_distance,
                                "steps": step,
                                "invalid_outputs": invalid_outputs,
                                "invalid_output_samples": invalid_samples,
                                "forced_max_step_stop": int(forced_stop),
                                "predicted_action_counts": dict(sorted(actions.items())),
                                "step_times_ms": step_times,
                                "episode_seconds": time.perf_counter() - started,
                                "trajectory_file": str(
                                    Path("trajectories") / _trajectory_name(scene_id, episode.episode_id)
                                ),
                            }
                            trajectory["outcome"] = {
                                "success": row["success"],
                                "spl": row["spl"],
                                "oracle_success": row["oracle_success"],
                                "navigation_error": row["navigation_error"],
                                "minimum_navigation_error": row["minimum_navigation_error"],
                                "steps": row["steps"],
                            }
                            trajectory_path = trajectory_dir / _trajectory_name(
                                scene_id, episode.episode_id
                            )
                            _write_atomic_json(trajectory_path, trajectory)
                            output.write(json.dumps(row, separators=(",", ":")) + "\n")
                            output.flush()
                            records.append(row)
                            completed.add(key)
                            episode_index += 1
                            running = summarize(records, identity)
                            print(json.dumps({
                                "event": "episode",
                                "checkpoint": checkpoint_name,
                                "episode": episode_index,
                                "total": len(expected_keys),
                                "scene": scene_id,
                                "episode_id": str(episode.episode_id),
                                "steps": step,
                                "success": row["success"],
                                "spl": row["spl"],
                                "running_success": running["success"],
                                "running_spl": running["spl"],
                                "invalid": invalid_outputs,
                            }), flush=True)
                            summary_path.write_text(json.dumps(running, indent=2) + "\n")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
            finally:
                if loader is not None:
                    loader.close()
                del session, policy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        env.close()

    summary = summarize(records, identity)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", **summary}), flush=True)
    return summary


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--policy", choices=("recurrent", "v0_uniform4", "v0_uniform8", "v1_sw4"), default="recurrent")
    parser.add_argument("--habitat-config", type=Path,
                        default=root / "configs/eval/vln_r2r.yaml")
    parser.add_argument("--data-path", type=Path, required=True,
                        help="R2R path template containing {split}")
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--output", type=Path, default=root / "evaluation/r2r")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--attention", choices=("flash_attention_2", "sdpa"),
                        default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Deterministic smoke prefix; 0 evaluates the complete split")
    parser.add_argument("--scenes", help="Optional comma-separated scene IDs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.max_steps < 1 or args.max_episodes < 0:
        parser.error("token/step limits must be positive and max-episodes nonnegative")
    args.checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    args.habitat_config = args.habitat_config.expanduser().resolve()
    args.data_path = args.data_path.expanduser().resolve()
    args.scenes_dir = args.scenes_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Full R2R evaluation requires CUDA")
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    for checkpoint in args.checkpoints:
        evaluate_checkpoint(args, checkpoint)


if __name__ == "__main__":
    main()
