"""Publish only a completed recurrent policy export, excluding optimizer shards."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--repo',required=True)
    args=parser.parse_args()
    run=Path(args.run);export=run/'export'
    if not (run/'TRAINING_COMPLETE').is_file():
        raise SystemExit('Upload requires a completed full-training marker')
    required=['navigation_config.json','prompt_protocol.json','run_manifest.json','memory.pt']
    if not all((export/name).is_file() for name in required) or not list((export/'backbone').glob('*.safetensors')):
        raise SystemExit('Incomplete policy export')
    # Keep authentication outside the repository and all training artifacts.
    token=(Path.home()/'.cache/stagevln/hf_token').read_text().strip()
    api=HfApi(token=token)
    api.create_repo(repo_id=args.repo,repo_type='model',private=True,exist_ok=True)
    api.upload_folder(repo_id=args.repo,repo_type='model',folder_path=str(export),
                      commit_message='Upload completed chronological R2R policy')
    (run/'HF_UPLOAD_COMPLETE').write_text(args.repo+'\n')
    print('Uploaded completed recurrent policy to '+args.repo)


if __name__=='__main__':main()
