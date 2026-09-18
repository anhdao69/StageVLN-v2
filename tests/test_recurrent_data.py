import hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from qwen_vl.data.episode_manifest import load_manifest
from qwen_vl.data.history import history_indices
from qwen_vl.train.losses import navigation_loss_per_state

class RecurrentDataTests(unittest.TestCase):
 def test_manifest_and_gap(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'episodes.jsonl'
   row=dict(episode='a',instruction='Go left.',frames=['R2R/train/a/step_0000_TURN_LEFT.png','R2R/train/a/step_0001_STOP.png'],actions=['TURN_LEFT','STOP'])
   p.write_text(json.dumps(row)+'\n');eps=load_manifest(p)
   with self.assertRaises(ValueError):load_manifest(p,expected_sha256='wrong')
   self.assertEqual(eps[0].frames[1].key.step,1)
   row['frames'][1]='R2R/train/a/step_0002_STOP.png';p.write_text(json.dumps(row)+'\n')
   with self.assertRaises(ValueError):load_manifest(p)
 def test_recent_across_boundaries(self):
  for t in range(32):self.assertEqual(history_indices(t,'recent',4),list(range(max(0,t-4),t+1)))
  self.assertEqual(history_indices(100,'current'),[100])
 def test_per_state_weight_and_grad(self):
  x=torch.randn(2,5,7,requires_grad=True);y=torch.tensor([[-100,1,2,3,4],[-100,-100,-100,-100,2]])
  a=navigation_loss_per_state(x,y)
  self.assertEqual(a.shape,(2,));a.mean().backward();self.assertTrue(torch.isfinite(x.grad).all())
