"""Load the audited full-episode JSONL, with no shuffled-step reconstruction."""
import hashlib
import json
import re
from pathlib import Path
from qwen_vl.contracts import EpisodeRecord, FrameRecord, FrameKey

ACTIONS = ('MOVE_FORWARD','TURN_LEFT','TURN_RIGHT','STOP')

def load_manifest(path, expected_sha256=None):
    path=Path(path).expanduser()
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('Episode manifest fingerprint mismatch')
    episodes=[];seen=set()
    with path.open() as handle:
        for line in handle:
            row=json.loads(line);eid=str(row['episode']);instruction=row['instruction']
            if eid in seen or not isinstance(instruction,str) or not instruction.strip():
                raise ValueError('Duplicate episode or empty instruction')
            seen.add(eid)
            if not row['frames'] or len(row['frames']) != len(row['actions']):
                raise ValueError('Frame/action length mismatch')
            frames=[]
            for t,(name,action) in enumerate(zip(row['frames'],row['actions'])):
                p=Path(name);match=re.fullmatch(r'step_(\d+)_(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\.png',p.name)
                if p.is_absolute() or '..' in p.parts or p.parent.name != eid:
                    raise ValueError('Noncanonical/cross-episode frame path')
                if not match or int(match[1])!=t or (action is not None and (action not in ACTIONS or match[2]!=action)):
                    raise ValueError('Missing/reordered frame or conflicting action')
                frames.append(FrameRecord(FrameKey('r2r',eid,t),name,action,f'{eid}/{p.name}'))
            episodes.append(EpisodeRecord(eid,instruction,hashlib.sha256(instruction.encode()).hexdigest(),tuple(frames)))
    if not episodes:raise ValueError('Empty episode manifest')
    return tuple(episodes)
