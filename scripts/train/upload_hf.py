"""Publish only the final model export after a successful training run."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--repo',required=True)
a=p.parse_args();folder=Path(a.folder)
if not (folder/'TRAINING_COMPLETE').is_file():
    raise SystemExit('Refusing upload: no successful-training marker')
if not list(folder.glob('*.safetensors')):
    raise SystemExit('Refusing upload: final safetensors weights missing')
token=(Path.home()/'.cache/stagevln/hf_token').read_text().strip()
api=HfApi(token=token)
api.create_repo(repo_id=a.repo,repo_type='model',private=True,exist_ok=True)
api.upload_folder(repo_id=a.repo,repo_type='model',folder_path=str(folder),
    allow_patterns=['*.safetensors','*.json','*.jinja','*.txt','*.md','*.model','merges.txt','vocab.json'],
    ignore_patterns=['checkpoint-*/**','source_snapshot/**','training_args.bin'],commit_message='Upload completed R2R training export')
(folder/'HF_UPLOAD_COMPLETE').write_text(a.repo+'\n')
print('Uploaded final model to '+a.repo)
