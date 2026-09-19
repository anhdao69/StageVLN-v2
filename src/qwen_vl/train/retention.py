"""Prune only complete periodic checkpoints; epoch checkpoints are permanent."""
import json
from pathlib import Path
import re
import shutil


def prune_steps(output, epoch, keep=5):
    if keep < 0:
        raise ValueError('keep must be nonnegative')
    paths = []
    for path in Path(output).glob(f'step-epoch-{epoch}-*'):
        match = re.fullmatch(rf'step-epoch-{epoch}-(\d+)', path.name)
        if match and not path.is_symlink() and (path/'manifest.json').is_file():
            # A parseable completion manifest is written last by save_checkpoint.
            json.loads((path/'manifest.json').read_text())
            paths.append((int(match[1]), path))
    paths.sort()
    for _, path in paths[:max(0, len(paths)-keep)]:
        shutil.rmtree(path)
