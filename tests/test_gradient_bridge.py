import unittest,copy
import torch
from qwen_vl.train.gradient_bridge import GradientBridge
from qwen_vl.train.episode_trainer import EpisodeTrainer
class BridgeTests(unittest.TestCase):
 def test_empty_reader_microbatch_is_rejected(self):
  for value in (0,-1):
   with self.assertRaises(ValueError):EpisodeTrainer(None,[],None,reader_microbatch=value)
 def test_shared_and_ancestor_outputs(self):
  torch.manual_seed(6);a=torch.nn.Linear(3,3).double();b=copy.deepcopy(a)
  x=torch.randn(2,3,dtype=torch.double)
  h=a(x);later=a(h);loss=(later.square().sum()+h.square().sum()+2*later.sum());loss.backward()
  h2=b(x);later2=b(h2);bridge=GradientBridge();p=bridge.leaf(h2);q=bridge.leaf(later2)
  self.assertIs(q,bridge.leaf(later2))
  q.square().sum().backward();p.square().sum().backward();(2*q.sum()).backward();bridge.backward()
  for lhs,rhs in zip(a.parameters(),b.parameters()):torch.testing.assert_close(lhs.grad,rhs.grad)
