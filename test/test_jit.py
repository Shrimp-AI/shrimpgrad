import unittest
from shrimpgrad import Tensor
from shrimpgrad.engine.jit import ShrimpJit
import numpy as np

import time

def _simple_test(add, extract=lambda x: x, N=10):
  for i in range(5):
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    s0 = time.perf_counter()
    c = add(a, b)
    e0 = time.perf_counter() - s0
    np.testing.assert_allclose(extract(c).numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    print(f"run {i+1} time={e0*1000}ms")

class TestJit(unittest.TestCase):
  def test_basic1(self):
    @ShrimpJit
    def f(x,y):
      z = x + y
      return z.realize()
    _simple_test(f)

  def test_simple_jit_reset(self):
    @ShrimpJit
    def add(a, b): return (a+b).realize()
    _simple_test(add)
    add.reset()
    _simple_test(add, N=20)

  def test_simple_jit_norealize(self):
    @ShrimpJit
    def add(a, b): return (a+b)
    _simple_test(add)

  def test_jit_multiple_outputs(self):
    @ShrimpJit
    def f(a, b): return (a+b).realize(), (a-b).realize(), (a*b).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c, d, e = f(a, b)
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(d.numpy(), a.numpy()-b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(e.numpy(), a.numpy()*b.numpy(), atol=1e-4, rtol=1e-5)