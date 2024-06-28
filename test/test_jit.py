import unittest

from shrimpgrad import Tensor
from shrimpgrad.engine.jit import ShrimpJit
import numpy as np


class TestJit(unittest.TestCase):
  def test_basic1(self):
    @ShrimpJit
    def f(x,y):
      z = x + y
      return z.realize()

    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))

    f(x,y)
    self.assertEqual(len(f.jit_kernels), 0)
    z = f(x,y)
    self.assertEqual(len(f.jit_kernels), 1)
    z = f(x,y)
    np.testing.assert_allclose(z.data(), 2.0)

