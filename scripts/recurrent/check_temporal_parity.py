"""Direct K8 vs bridged actual-Qwen gradients, and first-update memory credit."""
import argparse,copy,json
import torch
from types import SimpleNamespace
from transformers import AutoTokenizer
from check_adapter import tiny_model,BASE
from qwen_vl.contracts import EpisodeRecord,FrameRecord,FrameKey
from qwen_vl.models.navigation_policy import NavigationPolicy
from qwen_vl.train.episode_trainer import EpisodeTrainer

class Loader:
 def __init__(self):
  generator=torch.Generator().manual_seed(11)
  self.pixels=[torch.randn(16,1536,generator=generator) for _ in range(12)]
 def get(self,frame):return self.pixels[frame.key.step],torch.tensor([1,4,4])
 def prefetch(self,frames):pass

def run(reader_microbatch=2):
 torch.set_num_threads(1);torch.manual_seed(4)
 tok=AutoTokenizer.from_pretrained(BASE)
 base=NavigationPolicy(tiny_model(),tok,True,dict(slots=4,width=16,heads=2,ffn_width=32,layers=3))
 frames=tuple(FrameRecord(FrameKey('r2r','0',i),'unused','TURN_LEFT' if i<11 else 'STOP',str(i)) for i in range(12))
 episodes=[EpisodeRecord('0','Turn left at the door.','',frames)]
 report=[]
 for recent in (0,4):
  direct=copy.deepcopy(base);bridged=copy.deepcopy(base)
  # Helpers are plain Python objects; deepcopy must retain identity to the owned backbone.
  assert direct.reader.backbone is direct.backbone
  trainers=[EpisodeTrainer(direct,episodes,Loader(),recent,'direct',1,False),EpisodeTrainer(bridged,episodes,Loader(),recent,'bridge',reader_microbatch,False)]
  results=[]
  for trainer in trainers:
   trainer.policy.train();results.append(trainer.segment(SimpleNamespace(episode_index=0,steps=tuple(range(8)),first=True,last=False),1/8))
  # Batch GEMM/recurrent accumulation order can change FP32 rounding.
  torch.testing.assert_close(torch.tensor(results[0]['loss_sum']/8),torch.tensor(results[1]['loss_sum']/8),rtol=1e-5,atol=1e-5)
  errors={};maximum=0.
  for (name,a),(other,b) in zip(direct.named_parameters(),bridged.named_parameters()):
   assert name==other
   if a.grad is None or b.grad is None:assert a.grad is None and b.grad is None;continue
   torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=2e-5)
   maximum=max(maximum,float((a.grad-b.grad).abs().max()))
  for prefix in ['writer.initial_slots','writer.blocks.0','instruction_encoder','visual_projector','memory_adapter','backbone.model.visual.merger']:
   total=sum(float(p.grad.abs().sum()) for n,p in direct.named_parameters() if n.startswith(prefix) and p.grad is not None)
   assert total>0,prefix;errors[prefix]=total
  torch.testing.assert_close(trainers[0].carry['memory'],trainers[1].carry['memory'],rtol=1e-5,atol=1e-6)
  assert trainers[0].carry['memory'].grad_fn is None
  report.append(dict(recent=recent,states=8,reader_microbatch=reader_microbatch,loss=results[0]['loss_sum']/8,max_gradient_error=maximum,gradient_l1=errors))
 print(json.dumps(report,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--reader-microbatch',type=int,default=2)
 run(parser.parse_args().reader_microbatch)
