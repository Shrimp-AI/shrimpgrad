import unittest

from shrimpgrad.engine.scheduler2 import FusedKernelBuilder
from shrimpgrad.tensor import Tensor

class TestFusedKernelBuilder(unittest.TestCase):

  def test_one(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum().expand(10,10)
    d = c / b
    e = d.mean()

    fkb = FusedKernelBuilder(e.thunk)
    self.assertEqual(1, len(fkb.expands))
    self.assertEqual(2, len(fkb.loads))
    self.assertEqual(1, len(fkb.consts))
    self.assertEqual(2, len(fkb.fused_ops))
    self.assertEqual(2, len(fkb.unfused))

    fkb.schedule_loads()
    fkb.schedule_consts()
    from pprint import pprint
    pprint(fkb.scheduled_kernels)