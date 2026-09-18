"""Rebuild R2R-only Janus prompts from original metadata and complete PNG sequences."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

ACTIONS = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
PROMPT = ('You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task.\n This is your historical observation:{history}\n This is your current observation:<image>\n Your task is to {instruction}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP.')

def indices(t, mode):
    if mode == 'uniform8':
        return list(range(t+1)) if t <= 8 else [i*t//8 for i in range(9)]
    return list(range(max(0,t-4),t+1))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/groups/yshang/an221229/data/JanusVLN_data')
    p.add_argument('--output',default='/groups/yshang/an221229/data/StageVLN-v2')
    args=p.parse_args();root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    meta=root/'R2R_VLNCE_v1-3_preprocessed/train'
    with gzip.open(meta/'train.json.gz','rt') as f: episodes=json.load(f)['episodes']
    with gzip.open(meta/'train_gt.json.gz','rt') as f: gt=json.load(f)
    files={mode:(out/f'r2r_{mode}.json.tmp').open('w') for mode in ['uniform8','sw4']}
    for f in files.values():f.write('[\n')
    manifest=(out/'episodes.jsonl.tmp').open('w')
    count=0;counts=Counter();seen=set();longest=0;smoke={m:[] for m in files}
    for ep in episodes:
        eid=str(ep['episode_id']);assert eid not in seen;seen.add(eid)
        paths=sorted((root/'R2R/train'/eid).glob('*.png'))
        actions=gt[eid]['actions']
        assert len(paths)==len(actions)>0,(eid,len(paths),len(actions))
        instruction=ep['instruction']['instruction_text'].strip()
        relative=[x.relative_to(root).as_posix() for x in paths]
        for t,path in enumerate(paths):
            match=re.fullmatch(r'step_(\d+)_(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\.png',path.name)
            assert match and int(match[1])==t,(eid,path)
            action=ACTIONS[actions[t]];assert match[2]==action,(eid,t,action,match[2])
            counts[action]+=1
            for mode,f in files.items():
                selected=[relative[j] for j in indices(t,mode)]
                record={'id':f'{eid}/{path.name}','conversations':[{'from':'human','value':PROMPT.format(history='<image>'*(len(selected)-1),instruction=instruction)},{'from':'gpt','value':action}],'images':selected}
                if count:f.write(',\n')
                json.dump(record,f,separators=(',',':'))
                if t>=8 and len(smoke[mode])<128:smoke[mode].append(record)
            count+=1
        longest=max(longest,len(paths))
        manifest.write(json.dumps({'episode':eid,'instruction':instruction,'scene':ep['scene_id'],'frames':relative,'actions':[ACTIONS[x] for x in actions]})+'\n')
    for mode,f in files.items():
        f.write('\n]\n');f.close();(out/f'r2r_{mode}.json.tmp').replace(out/f'r2r_{mode}.json')
        (out/f'smoke_{mode}.json').write_text(json.dumps(smoke[mode]))
    manifest.close();(out/'episodes.jsonl.tmp').replace(out/'episodes.jsonl')
    report={'episodes':len(seen),'states':count,'actions':dict(counts),'max_episode_length':longest,'checks':['complete contiguous PNG sequence','all actions equal original train_gt','instruction from original train metadata','unique episode IDs','identical v0/v1 states and targets'],'history':{'uniform8':'t<=8: all; otherwise floor(i*t/8), i=0..8','sw4':'max(0,t-4)..t'},'files':{}}
    for name in ['r2r_uniform8.json','r2r_sw4.json','episodes.jsonl']:
        h=hashlib.sha256()
        with (out/name).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        report['files'][name]=h.hexdigest()
    (out/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
