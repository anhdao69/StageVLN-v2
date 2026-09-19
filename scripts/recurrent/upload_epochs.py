"""Upload immutable completed epoch exports while training continues."""
import argparse
import json
from pathlib import Path
import time
from huggingface_hub import HfApi


def upload_epoch(api, run, epoch, prefix):
    directory = run/f'epoch-{epoch}'
    marker = directory/'EPOCH_COMPLETE'
    if not marker.is_file():
        return False
    repo = f'{prefix}-epoch{epoch}'
    done = directory/'HF_UPLOAD_COMPLETE'
    if done.exists():
        if done.read_text().strip() != repo:
            raise ValueError('Upload destination changed')
        return True
    export = directory/'export'
    manifest = json.loads((export/'run_manifest.json').read_text())
    if manifest['max_episodes'] or json.loads(marker.read_text())['epoch'] != epoch:
        raise ValueError('Refusing smoke or inconsistent epoch export')
    required = ['navigation_config.json', 'prompt_protocol.json', 'memory.pt']
    if not all((export/name).is_file() for name in required) or not list((export/'backbone').glob('*.safetensors')):
        raise ValueError('Incomplete policy export')
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(export),
                      delete_patterns=['training-status.txt'],
                      commit_message=f'Completed R2R epoch {epoch}')
    temporary = done.with_suffix('.tmp')
    temporary.write_text(repo+'\n')
    temporary.replace(done)
    print('Uploaded '+repo, flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    run = Path(args.run)
    api = HfApi(token=(Path.home()/'.cache/stagevln/hf_token').read_text().strip())
    failures = 0
    while True:
        try:
            finished = [upload_epoch(api, run, epoch, args.prefix) for epoch in range(1, args.epochs+1)]
            if all(finished):
                return
            if (run/'TRAINING_PROCESS_EXITED').exists():
                raise SystemExit('Training exited before all epoch exports completed')
            failures = 0
        except Exception as error:
            failures += 1
            # Do not print HTTP request headers or credentials.
            print(f'Upload attempt failed ({type(error).__name__}); retry {failures}/10', flush=True)
            if failures >= 10:
                raise SystemExit('Epoch upload failed after ten retries; exports remain on disk')
        time.sleep(30)


if __name__ == '__main__':
    main()
