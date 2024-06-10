import unittest

from shrimpgrad.engine.lower import LowerFusedKernel
from shrimpgrad.engine.scheduler import FusedKernelBuilder, print_schedule
from shrimpgrad.tensor import Tensor

class TestPython(unittest.TestCase):
  def test_basic(self):
    x = Tensor.rand(2,2)
    y = Tensor.rand(2,2)
    out = x + y

    fkb = FusedKernelBuilder(out.thunk)
    schedule = fkb.schedule()
    print_schedule(schedule)
    self.assertEqual(3, len(schedule))
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    for i, ir_graph in enumerate(ir_graphs):
      print(f"Graph {i}")
      ir_graph.print()


