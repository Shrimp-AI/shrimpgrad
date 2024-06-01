import unittest

from shrimpgrad.future import IndexedForwardGraph
from shrimpgrad.tensor import Tensor
from shrimpgrad.engine.fuse_ops import FusionEngine


class TestFuseOps(unittest.TestCase):
  
  def test_basic_no_fusion(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    g = IndexedForwardGraph(a.thunk)
    fusion = FusionEngine(g)
    groups, _ = fusion.fuse()
    self.assertEqual(len(groups), 0)

  def test_basic_fuse2(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    g = IndexedForwardGraph(b.thunk)
    fusion = FusionEngine(g)
    fused, _ = fusion.fuse()
    self.assertEqual(len(fused), 1)


  def test_basic_fuse3(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum()
    g = IndexedForwardGraph(c.thunk)
    fusion = FusionEngine(g)
    fused_ops, _ = fusion.fuse()  
    self.assertEqual(len(fused_ops), 1)

  def test_two_fusions(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum().expand(10,10)
    d = c / b
    e = d.mean()
    g = IndexedForwardGraph(e.thunk)
    fusion = FusionEngine(g)
    fused_ops, _ = fusion.fuse()
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
    g = IndexedForwardGraph(e.thunk)
    fusion = FusionEngine(g)
    fused_ops, _ = fusion.fuse()
    self.assertEqual(1, len(fused_ops))


    
