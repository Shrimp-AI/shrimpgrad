from typing import Callable, List
import unittest
from shrimpgrad.engine.graph import log_thunk
import shrimpgrad.nn as nn

from shrimpgrad.engine.scheduler import FusedKernelBuilder, print_schedule
from shrimpgrad.runtime.ops import LoadOps
from shrimpgrad.tensor import Tensor

class TestFusedKernelBuilder(unittest.TestCase):
  # TODO: Make these test actually assert something and remove the prints
  def test_one(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum().expand(10,10)
    d = c / b
    e = d.mean()
    log_thunk(e.thunk)
    fkb = FusedKernelBuilder(e.thunk)
    print_schedule(fkb.schedule())

  def test_big_net(self): 
    class Model:
      def __init__(self):
        self.layers: List[Callable[[Tensor], Tensor]] = [
        nn.Linear(2, 50), Tensor.relu,
        nn.Linear(50, 50), Tensor.relu,
        nn.Linear(50, 50), Tensor.relu,
        nn.Linear(50, 1), Tensor.sigmoid, 
        ]
      def __call__(self, x: Tensor):
        return x.sequential(self.layers)

    x = Tensor.randn(100,2)
    model = Model()
    out = model(x)
    log_thunk(out.thunk)
    fkb = FusedKernelBuilder(out.thunk)
    print_schedule(fkb.schedule())
  
  def test_diamond_schedule(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y

    b = z * a 

    c = w * a 

    d = b / c

    out = d.sum()
    
    fkb = FusedKernelBuilder(out.thunk)
    print_schedule(fkb.schedule())
  
  def test_load_const_nd(self):
    out = Tensor.full((2,2,2), 3.0)
    fkb = FusedKernelBuilder(out.thunk)
    sched = fkb.schedule()
    self.assertEqual(1, len(sched))
    load_const_kernel = sched[0]
    inp = load_const_kernel.computation.ins[0]
    # No input buffers for a load const
    self.assertEqual([], inp)
    self.assertEqual(LoadOps.CONST, load_const_kernel.computation.ops[0])
    self.assertEqual(3.0, load_const_kernel.computation.args[0])
    self.assertEqual(32, (mbuff:=load_const_kernel.computation.out[0]).buff.nbytes)
    self.assertEqual((2,2,2), mbuff.vt.shape)