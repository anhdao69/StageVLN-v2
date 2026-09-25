import json
import hashlib

from qwen_vl.data.data_qwen import QWEN3_5_NON_THINKING_CHAT_TEMPLATE
from qwen_vl.eval.habitat_r2r import (
    HABITAT_ACTIONS, _trajectory_name, summarize, validate_export,
    validate_v0_export,
    validate_v1_export,
)


def test_habitat_action_contract_and_summary():
    assert HABITAT_ACTIONS == {
        "STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3,
    }
    identity = {"checkpoint": "fixture"}
    rows = [
        dict(identity=identity, success=1, spl=.5, oracle_success=1,
             navigation_error=2, steps=2, invalid_outputs=0,
             forced_max_step_stop=0, predicted_action_counts={"MOVE_FORWARD": 2},
             step_times_ms=[1, 3]),
        dict(identity=identity, success=0, spl=0, oracle_success=1,
             navigation_error=4, steps=1, invalid_outputs=1,
             forced_max_step_stop=1, predicted_action_counts={"STOP": 1},
             step_times_ms=[2]),
    ]
    result = summarize(rows, identity)
    assert result["episodes"] == 2
    assert result["success"] == .5 and result["spl"] == .25
    assert result["oracle_success"] == 1 and result["navigation_error"] == 3
    assert result["total_steps"] == 3 and result["invalid_outputs"] == 1
    assert result["predicted_action_counts"] == {"MOVE_FORWARD": 2, "STOP": 1}
    assert result["latency_ms"]["mean"] == 2
    assert _trajectory_name("scene/a", "episode:1") == "scene_a__episode_1.json"


def test_export_validation_matches_v2_and_v3_reader_windows(tmp_path):
    (tmp_path / "backbone").mkdir()
    for name in ("config.json", "processor_config.json", "tokenizer.json",
                 "model.safetensors.index.json"):
        (tmp_path / "backbone" / name).write_text("{}")
    (tmp_path / "memory.pt").write_bytes(b"fixture")
    (tmp_path / "navigation_config.json").write_text(json.dumps({
        "memory_enabled": True,
        "memory": {"slots": 64, "width": 512, "layers": 3,
                   "heads": 8, "ffn_width": 2048},
        "recent": 4,
    }))
    prompt = {"actions": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"],
              "chat_template": QWEN3_5_NON_THINKING_CHAT_TEMPLATE,
              "recent": 4}
    (tmp_path / "prompt_protocol.json").write_text(json.dumps(prompt))
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "config": {"variant": "v3_mem64_r4", "recent": 4,
                   "memory_enabled": True},
        "memory_architecture": {"slots": 64, "width": 512, "layers": 3,
                                "heads": 8, "ffn_width": 2048},
        "max_episodes": 0
    }))
    assert validate_export(tmp_path)["navigation"]["recent"] == 4
    navigation = json.loads((tmp_path / "navigation_config.json").read_text())
    navigation["recent"] = 0
    (tmp_path / "navigation_config.json").write_text(json.dumps(navigation))
    try:
        validate_export(tmp_path)
    except ValueError as error:
        assert "Reader window disagrees" in str(error)
    else:
        raise AssertionError("Mismatched reader window was accepted")
    prompt["recent"] = 0
    (tmp_path / "prompt_protocol.json").write_text(json.dumps(prompt))
    run = json.loads((tmp_path / "run_manifest.json").read_text())
    run["config"]["recent"] = 0
    run["config"]["variant"] = "v2_mem64_r0"
    (tmp_path / "run_manifest.json").write_text(json.dumps(run))
    assert validate_export(tmp_path)["navigation"]["recent"] == 0


def test_v0_export_validation_requires_uniform4_metadata(tmp_path):
    for name in ("config.json", "processor_config.json", "tokenizer.json",
                 "model.safetensors.index.json"):
        (tmp_path / name).write_text(json.dumps({"model_type": "qwen3_5"}))
    prompt = {"actions": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"],
              "chat_template": QWEN3_5_NON_THINKING_CHAT_TEMPLATE,
              "template_sha256": hashlib.sha256(QWEN3_5_NON_THINKING_CHAT_TEMPLATE.encode()).hexdigest(),
              "system": "You are a helpful assistant.",
              "assistant_prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
              "target_suffix": "<|im_end|>\n",
              "dataset_config": {"history": "uniform4"}}
    (tmp_path / "prompt_protocol.json").write_text(json.dumps(prompt))
    (tmp_path / "training_metadata.json").write_text(json.dumps({"variant": "v0_uniform4"}))
    assert validate_v0_export(tmp_path)["training"]["variant"] == "v0_uniform4"
    prompt["dataset_config"]["history"] = "uniform8"
    (tmp_path / "prompt_protocol.json").write_text(json.dumps(prompt))
    try:
        validate_v0_export(tmp_path)
    except ValueError as error:
        assert "Uniform4" in str(error)
    else:
        raise AssertionError("Uniform8 metadata was accepted for the v0 Uniform4 evaluator")


def test_v0_export_validation_requires_uniform8_metadata(tmp_path):
    for name in ("config.json", "processor_config.json", "tokenizer.json",
                 "model.safetensors.index.json"):
        (tmp_path / name).write_text(json.dumps({"model_type": "qwen3_5"}))
    prompt = {"actions": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"],
              "chat_template": QWEN3_5_NON_THINKING_CHAT_TEMPLATE,
              "template_sha256": hashlib.sha256(QWEN3_5_NON_THINKING_CHAT_TEMPLATE.encode()).hexdigest(),
              "system": "You are a helpful assistant.",
              "assistant_prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
              "target_suffix": "<|im_end|>\n",
              "dataset_config": {"history": "uniform8"}}
    (tmp_path / "prompt_protocol.json").write_text(json.dumps(prompt))
    (tmp_path / "training_metadata.json").write_text(json.dumps({"variant": "v0_uniform8"}))
    assert validate_v0_export(tmp_path, "uniform8")["training"]["variant"] == "v0_uniform8"
    prompt["dataset_config"]["history"] = "uniform4"
    (tmp_path / "prompt_protocol.json").write_text(json.dumps(prompt))
    try:
        validate_v0_export(tmp_path, "uniform8")
    except ValueError as error:
        assert "Uniform8" in str(error)
    else:
        raise AssertionError("Uniform4 metadata was accepted for the v0 Uniform8 evaluator")


def test_v1_export_validation_requires_sliding_window_four(tmp_path):
    for name in ("config.json", "processor_config.json", "tokenizer.json",
                 "model.safetensors.index.json"):
        (tmp_path / name).write_text(json.dumps({"model_type": "qwen3_5"}))
    prompt = {"actions": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"],
              "chat_template": QWEN3_5_NON_THINKING_CHAT_TEMPLATE,
              "template_sha256": hashlib.sha256(QWEN3_5_NON_THINKING_CHAT_TEMPLATE.encode()).hexdigest(),
              "system": "You are a helpful assistant.",
              "assistant_prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
              "target_suffix": "<|im_end|>\n",
              "dataset_config": {"history": "sw4"}}
    (tmp_path / "prompt_protocol.json").write_text(json.dumps(prompt))
    assert validate_v1_export(tmp_path)["prompt"]["dataset_config"]["history"] == "sw4"
    prompt["dataset_config"]["history"] = "uniform4"
    (tmp_path / "prompt_protocol.json").write_text(json.dumps(prompt))
    try:
        validate_v1_export(tmp_path)
    except ValueError as error:
        assert "Sliding Window 4" in str(error)
    else:
        raise AssertionError("Uniform4 metadata was accepted for the v1 Sliding Window 4 evaluator")
