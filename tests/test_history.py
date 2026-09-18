import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('prepare',Path(__file__).parents[1]/'scripts/data/prepare_r2r.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class HistoryTests(unittest.TestCase):
 def test_recent_is_not_uniform_tail(self):
  self.assertEqual(module.indices(100,'sw4'),[96,97,98,99,100])
  self.assertEqual(module.indices(100,'uniform8'),[0,12,25,37,50,62,75,87,100])
 def test_all_steps(self):
  for t in range(513):
   for mode in ['sw4','uniform8']:
    x=module.indices(t,mode)
    self.assertEqual(x,sorted(set(x)))
    self.assertEqual(x[-1],t)
    self.assertGreaterEqual(x[0],0)
    self.assertEqual(len(x),min(t+1,5 if mode=='sw4' else 9))
