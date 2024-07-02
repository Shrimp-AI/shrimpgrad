import ctypes
import unittest
from shrimpgrad import Tensor
from shrimpgrad.engine.jit import ShrimpJit
from shrimpgrad.nn.optim import SGD
import numpy as np


import time

from shrimpgrad.nn import Linear, get_parameters

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
      a = Tensor.randn(9, 10)
      b = Tensor.randn(9, 10)
      c, d, e = f(a, b)
      np.testing.assert_allclose(c.data(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(d.data(), a.numpy()-b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(e.data(), a.numpy()*b.numpy(), atol=1e-4, rtol=1e-5)

  def test_jit_multiple_outputs_const(self):
    @ShrimpJit
    def f(a, b): return (a+b).realize(), (a-b).realize(), (a*b).realize()
    for _ in range(5):
      a = Tensor.rand()
      b = Tensor.rand()
      c, d, e = f(a, b)
      print(f"{a.data() = } {b.data() = }")
      print(f"{c.data() = } {d.data() = } {e.data() = }")
      np.testing.assert_allclose(c.data(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(d.data(), a.numpy()-b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(e.data(), a.numpy()*b.numpy(), atol=1e-4, rtol=1e-5)

  def test_jit_baby_nn(self):
    class Model:
      def __init__(self):
        self.w = Tensor.ones((2,2))
      def __call__(self, x: Tensor):
        return (x + self.w).relu()

    m = Model()
    x = Tensor.ones((2,2))
    sgd = SGD([m.w], lr=1.0)
    def f(x):
      out = m(x)
      loss = out.mean()
      loss.backward()
      sgd.step()
      return loss

    f(x)
    f(x)
    f(x)
    f(x)

    out = m(x)
    out.realize()

  def test_jit_nn_model(self):
    class Model:
      def __init__(self):
        self.layers = [
          Linear(10,10), Tensor.relu,
          Linear(10,1), Tensor.relu
        ]
      def __call__(self, x: Tensor):
        return x.sequential(self.layers)
      def layer_one_weights(self):
        return self.layers[0].w.thunk.base.buff._pointer(ctypes.c_float)[0:10], self.layers[0].bias
    m = Model()
    y = Tensor.ones((10,1))
    x = Tensor.ones((10,10))
    sgd = SGD(get_parameters(m), lr=1.0)
    @ShrimpJit
    def f(x):
      out = m(x)
      loss = out.sub(y).square().mean()
      loss.backward()
      sgd.step()
      return loss
    for _ in range(5): f(x)
    _ = m(x).realize()
  
  def test_wah(self):
    x = Tensor.ones((2,2))
    y = Tensor.full((2,2), 3.0)
    class Model:
      def __init__(self):
        self.p1 = Tensor.full((2,2), 2.0, requires_grad=True)
        self.p2 = Tensor.full((2,2), 3.0, requires_grad=True)
      def __call__(self, x,y): 
        return (y/self.p1).sum()
    m = Model() 
    sgd = SGD([m.p1],lr=0.0000001)
    @ShrimpJit
    def train_step(x,y):
      sgd.zero_grad()
      out = m(x,y)
      out.backward()
      sgd.step()
      return out
    from shrimpgrad.engine.graph import log_thunk
    for _ in range(3):
      out = train_step(x,y)
      print(out.data()) 
      print(m.p1.grad.data())
      # print(m.p2.grad.data())
      print("NEXT STEP")
    log_thunk(out.thunk)
