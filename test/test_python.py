import unittest

from shrimpgrad.engine.lower import LowerFusedKernel
from shrimpgrad.engine.scheduler import FusedKernelBuilder
from shrimpgrad.tensor import Tensor
from shrimpgrad.runtime.python import PyCodeGen

class TestPython(unittest.TestCase):
  def test_basic(self):
    x = Tensor.rand(2,2)
    y = Tensor.rand(2,2)
    out = x + y
    fkb = FusedKernelBuilder(out.thunk)
    schedule = fkb.schedule()
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    pcg = PyCodeGen(ir_graphs)
    pcg.gen()
    pcg.print()

  def test_linear_model(self):
    x = Tensor.ones((10,20))
    w = Tensor.ones((10,20))
    b = Tensor.ones((10,))
    out = x.dot(w.transpose())+b
    out.realize()
    print(out.data())


