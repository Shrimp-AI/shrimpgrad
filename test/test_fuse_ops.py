import unittest

from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.tensor import Tensor
from shrimpgrad.engine.fuse_ops import bfs, FusionEngine


class TestFuseOps(unittest.TestCase):
  def test_basic_elementiwse_fuse(self):
    _ = Tensor.randn(10,10)
    # # LoadOps.EMPTY -> LoadOps.COPY
    # y = Tensor.randn(10,10)
  
  def test_bfs_grouping(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y
    b = a * z
    c = b / w
    out = b - c

    fusion = FusionEngine(out.thunk)
    self.assertEqual(1, len(fusion.real_roots))
    self.assertEqual(fusion.real_roots[0], a.thunk)
  
  def test_basic_no_fusion(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    log_thunk(a.thunk)
    fusion = FusionEngine(a.thunk)
    fused_ops = fusion.start()
    self.assertEqual(0, len(fused_ops))

  def test_basic_fuse2(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    log_thunk(b.thunk)
    fusion = FusionEngine(b.thunk)
    fused_ops = fusion.start()
    self.assertEqual(1, len(fused_ops))
    self.assertEqual(2, len(fused_ops.injectives))

  def test_basic_fuse3(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum()
    log_thunk(c.thunk)
    fusion = FusionEngine(c.thunk)
    fused_ops = fusion.start()
    self.assertEqual(1, len(fused_ops))
    self.assertEqual(2, len(fused_ops[0].injectives))
    self.assertTrue(fused_ops[0].has_reduce)