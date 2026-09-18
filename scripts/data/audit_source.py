import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,'scripts/data')
from prepare_r2r import PROMPT,indices
root=Path('/groups/yshang/an221229/data/JanusVLN_data')
eps={r['episode']:r for r in map(json.loads,open('/groups/yshang/an221229/data/StageVLN-v2/episodes.jsonl'))}
def records(path):
 dec=json.JSONDecoder();buf='';pos=0;eof=False
 with open(path) as f:
  while True:
   if pos>len(buf)-4096 and not eof:
    buf=buf[pos:]+f.read(1024*1024);pos=0
   while pos<len(buf) and (buf[pos].isspace() or buf[pos] in '[,'):pos+=1
   if pos<len(buf) and buf[pos]==']':return
   try:r,end=dec.raw_decode(buf,pos)
   except json.JSONDecodeError:
    chunk=f.read(1024*1024)
    if not chunk:raise
    buf=buf[pos:]+chunk;pos=0;continue
   pos=end;yield r
seen=set();count=0;duplicate=0
for r in records('/home/an221229/code/SpatialForcing-VLN/train_r2r_rxr.json'):
 if '/R2R/train/' not in r['images'][-1]:continue
 eid,filename=r['id'].split('/');t=int(filename.split('_')[1]);ep=eps[eid]
 expected=[ep['frames'][j] for j in indices(t,'uniform8')]
 actual=['R2R/train/'+x.split('/R2R/train/',1)[1] for x in r['images']]
 assert actual==expected,(r['id'],'history')
 assert r['conversations']==[{'from':'human','value':PROMPT.format(history='<image>'*(len(expected)-1),instruction=ep['instruction'])},{'from':'gpt','value':ep['actions'][t]}],r['id']
 duplicate+=r['id'] in seen;seen.add(r['id']);count+=1
print(json.dumps(dict(source_r2r_records=count,unique_states=len(seen),duplicates=duplicate,expected_states=sum(len(ep['frames']) for ep in eps.values()),all_prompts_labels_uniform_indices_match=True),indent=2))
