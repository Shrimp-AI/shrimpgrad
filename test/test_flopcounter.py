import unittest

from shrimpgrad import Tensor
from shrimpgrad.engine.scheduler import FusedKernelBuilder, print_schedule
from shrimpgrad.engine.perf import flop_counter

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
    mul = x*y
    sum = mul.sum()
    add = sum + sum
    s = FusedKernelBuilder(add.thunk).schedule()
    print_schedule(s)
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
    y = Tensor.ones((2,2))
    op0 = x.sum(0)
    s = FusedKernelBuilder(op0.thunk).schedule()

    flop = flop_counter(s)
    assert flop == 16

    op1 = op0.sum(1)
    s = FusedKernelBuilder(op1.thunk).schedule()
    flop = flop_counter(s)
    assert flop ==  16+4
