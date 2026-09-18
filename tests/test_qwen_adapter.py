import unittest
import torch
from qwen_vl.models.qwen_adapter import premerge_to_raster
class FeatureOrderTests(unittest.TestCase):
 def test_rectangular_inverse(self):
  raster=torch.arange(6*10*3).reshape(1,6,10,3)
  rows=raster.reshape(1,3,2,5,2,3).permute(0,1,3,2,4,5).reshape(60,3)
  torch.testing.assert_close(premerge_to_raster(rows,(1,6,10),2),raster)
 def test_video_rejected(self):
  with self.assertRaises(ValueError):premerge_to_raster(torch.zeros(8,3),(2,2,2),2)
