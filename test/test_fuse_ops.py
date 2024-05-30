import unittest

from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.tensor import Tensor
from shrimpgrad.engine.fuse_ops import FusionEngine


class TestFuseOps(unittest.TestCase):
  
  def test_basic_no_fusion(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    log_thunk(a.thunk)
    fusion = FusionEngine(a.thunk)
    groups = fusion.fuse()
    self.assertEqual(len(groups), 0)

  def test_basic_fuse2(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    log_thunk(b.thunk)
    fusion = FusionEngine(b.thunk)
    fused = fusion.fuse()
    self.assertEqual(len(fused), 1)


  def test_basic_fuse3(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum()
    log_thunk(c.thunk)
    fusion = FusionEngine(c.thunk)
    fused_ops = fusion.fuse()  
    self.assertEqual(len(fused_ops), 1)

  def test_two_fusions(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum().expand(10,10)
    d = c / b
    e = d.mean()
    log_thunk(e.thunk)
    fusion = FusionEngine(e.thunk)
    fused_ops = fusion.fuse()
    self.assertEqual(2, len(fused_ops))
   
  def test_diamond_fuse(self): 
    
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y

    b = z * a 

    c = w * a 

    d = b / c

    e = d.sum()
    log_thunk(e.thunk)
    fusion = FusionEngine(e.thunk)
    fused_ops = fusion.fuse()
    self.assertEqual(1, len(fused_ops))


    
