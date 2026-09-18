import json,sys,random
from pathlib import Path
sys.path.insert(0,'scripts/data')
from prepare_r2r import PROMPT,indices
folder=Path('/groups/yshang/an221229/data/StageVLN-v2')
eps=list(map(json.loads,(folder/'episodes.jsonl').read_text().splitlines()))
eps.sort(key=lambda x:len(x['instruction']),reverse=True)
for mode in ['uniform8','sw4']:
 records=[]
 for ep in eps[:64]:
  for t in [0,len(ep['frames'])-1]:
   selected=[ep['frames'][i] for i in indices(t,mode)]
   records.append(dict(id=ep['episode']+'/'+Path(ep['frames'][t]).name,images=selected,conversations=[]))
   records[-1]['conversations']=[{'from':'human','value':PROMPT.format(history='<image>'*(len(selected)-1),instruction=ep['instruction'])},{'from':'gpt','value':ep['actions'][t]}]
 (folder/f'stress_{mode}.json').write_text(json.dumps(records))
 version='v0' if mode=='uniform8' else 'v1'
 cfg=json.loads(Path(f'configs/datasets/newton_r2r_{version}_smoke.json').read_text())
 cfg['annotation_path']=str(folder/f'stress_{mode}.json')
 Path(f'configs/datasets/newton_r2r_{version}_stress.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Wrote 128 states per version: longest instructions, mixed initial/final frames')
