import unittest

from shrimpgrad import Tensor

from shrimpgrad.engine.lower import LowerFusedKernel
from shrimpgrad.engine.scheduler import FusedKernelBuilder
from shrimpgrad.runtime.clang import ClangCodeGen, ClangDevice


class TestClang(unittest.TestCase):
  def test_add(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    z = x + y
    fkb = FusedKernelBuilder(z.thunk)
    schedule = fkb.schedule()
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    pcg = ClangCodeGen(ir_graphs)
    pcg.gen()

  def test_jitify_exists(self):
    x = Tensor.ones((2,2), device=ClangDevice())
    x.device.jitify([])

