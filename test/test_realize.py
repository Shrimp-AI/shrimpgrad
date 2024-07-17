# Test some quirks of realize

import unittest
import numpy as np

from shrimpgrad import Tensor
from shrimpgrad.nn import Linear, get_parameters


class TestRealize(unittest.TestCase):
  def test_basic_realize(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    z = x + y
    z.realize()
    assert x.thunk.realized
    assert y.thunk.realized
    assert z.thunk.realized

  def test_basic_realize_movement(self):
    x = Tensor.ones((2,2)).reshape(1,1,2,2).expand(2,2,2,2)
    y = Tensor.ones((2,2)).reshape(1,1,2,2).expand(2,2,2,2)

    assert x.thunk.base != x.thunk
    assert y.thunk.base != y.thunk

    z = x + y
    z.realize()

    np.testing.assert_allclose(z.data(), np.ones((2,2,2,2)) + np.ones((2,2,2,2)))

  def test_parameters_remain_realized_after_multiple_fwd_passes(self):
    class Model:
      def __init__(self):
        self.layers = [
          Linear(10,10), Tensor.relu,
          Linear(10,1), Tensor.relu
        ]
      def __call__(self, x: Tensor):
        return x.sequential(self.layers)

    m = Model()
    x = Tensor.ones((10,10))

    out = m(x)

    ps = get_parameters(m)

    for p in ps:
      assert not p.thunk.realized

    out.realize()

    for p in ps:
      assert p.thunk.realized

    out = m(x)

    assert len(ps) == 4

    ps = get_parameters(m)

    for p in ps:
      assert p.thunk.realized
      assert p.thunk.buff.allocated

    out.realize()
