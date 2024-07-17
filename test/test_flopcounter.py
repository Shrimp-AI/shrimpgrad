import unittest

from shrimpgrad import Tensor
from shrimpgrad.engine.scheduler import FusedKernelBuilder
from shrimpgrad.engine.perf import flop_counter, memory_estimator

class TestFlopCounter(unittest.TestCase):
  def test_single_add(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    z = x+y
    s = FusedKernelBuilder(z.thunk).schedule()
    flop = flop_counter(s)
    assert flop == 4

  def test_double_mul(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    z = x+y
    zz = z + y
    s = FusedKernelBuilder(zz.thunk).schedule()
    flop = flop_counter(s)
    assert flop == 8

  def test_mul_sum_add(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    m = x*y
    s= m.sum()
    a = s + s
    s = FusedKernelBuilder(a.thunk).schedule()
    flop = flop_counter(s)
    assert flop == 9

  def test_chained_add(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    out0 = x + y
    out1 = out0 + y
    out2 = out0 + out1
    s = FusedKernelBuilder(out2.thunk).schedule()
    flop = flop_counter(s)
    assert flop == 12

  def test_flops_sum2d(self):
    x = Tensor.ones((4,4))
    op0 = x.sum(0)
    s = FusedKernelBuilder(op0.thunk).schedule()

    flop = flop_counter(s)
    assert flop == 16

    op1 = op0.sum(1)
    s = FusedKernelBuilder(op1.thunk).schedule()
    flop = flop_counter(s)
    assert flop ==  16+4

  def test_memory_add(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    out0 = x + y
    s = FusedKernelBuilder(out0.thunk).schedule()
    mem = memory_estimator(s)
    assert mem == 80 