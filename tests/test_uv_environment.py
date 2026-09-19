"""The committed uv environment must be complete and reproducible."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
UV = shutil.which("uv")


@pytest.mark.skipif(UV is None, reason="uv is not installed")
def test_uv_lock_is_current_and_resolves_for_this_server():
    result = subprocess.run(
        [UV, "sync", "--locked", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(UV is None, reason="uv is not installed")
def test_uv_runtime_export_contains_validated_training_stack():
    result = subprocess.run(
        [UV, "export", "--locked", "--no-dev", "--no-emit-project", "--no-hashes"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    requirements = set(result.stdout.splitlines())
    for expected in {
        "accelerate==1.13.0",
        "deepspeed==0.16.4",
        "flash-attn==2.8.3+cu.12.9.torch.2.10",
        "torch==2.10.0+cu129",
        "torchvision==0.25.0+cu129",
        "transformers==5.3.0",
    }:
        assert expected in requirements
