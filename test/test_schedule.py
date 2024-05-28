import unittest

from shrimpgrad.nn import Linear 
from shrimpgrad import Tensor
from shrimpgrad.engine.schedule import Scheduler
from shrimpgrad.runtime.ops import BufferOps, LoadOps 
from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.engine.fusion import semidominator
from pprint import pprint

class ScheduleTest(unittest.TestCase):
  def test_schedule_basic(self):
    x = Tensor.full((2,2), 2.0)

    s = Scheduler([x.thunk])
    schedule = s.schedule()

    self.assertEqual(1, len(schedule))
    self.assertEqual(schedule[0].ast.op, LoadOps.COPY)
 
  def test_schedule_basic2(self):
    x = Tensor.full((2,2), 1.0)
    y = Tensor.full((2,2), 2.0)
    z = x + y

    # LoadCopy, LoadCopy
    # BufferStore -> add -> load copy, load copy

    s = Scheduler([z.thunk])
    sched = s.schedule()
    self.assertEqual(3, len(sched))
    self.assertEqual(sched[2].ast.op, BufferOps.STORE)
  
  def test_schedule_dot(self):
    x = Tensor.full((2,2), 1.0)
    y = Tensor.full((2,2), 2.0)

    # LoadCopy <- LoadEmpty, LoadCopy<-LoadEmpty
    # x.reshape(...)  Thunk with base pointer to x 
    # y.reshape(...).permute(...) Thunk with base pointer to y (needs multiple views to handle the permute)
    # z = x*y
    # z.sum(ax=-1) don't handle reduces yet

    z = x.dot(y)
    s = Scheduler([z.thunk])
    sched = s.schedule()
    self.assertEqual(3, len(sched))
    self.assertEqual(sched[0].ast.op, BufferOps.STORE)
    
  def test_schedule_basic_neural_net(self):
    # LoadOps.EMPTY -> LoadOps.COPY
    x = Tensor.randn(10,10)
    # LoadOps.EMPTY -> LoadOps.COPY
    y = Tensor((10,), [1,0,1,0,1,0,1,0,0,1])
    # Two Empty -> Copy for generating random weights and bias
    # z = x.dot(w.transpose()) + bias
    # x reshape, y reshape permute
    # BinaryMul, ReduceSum, BinaryAdd (bias)
    # loss = z.mse(y) -> (z-y)*(z-y).mean()
    # BinarySub, BinaryMul, ReduceSum, BinaryDiv 
    # BinaryDiv will have a reshape(...).expand(...) without a matching buffer (needs a const load or something)
    # Would expect 4 copy kernels, 1 const kernel?,  and 1 store kernel
    # 1 exec kernel
    out = Linear(10, 1)(x).mse(y)
    log_thunk(out.thunk)
    sched = Scheduler([out.thunk]).schedule()
    self.assertEqual(6, len(sched))

  def test_post_dominator(self):
    x = Tensor.randn(10,10)
    # LoadOps.EMPTY -> LoadOps.COPY
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y
    b = a * z
    c = b / w
    out = b - c

    log_thunk(out.thunk)
    sdom = semidominator(out.thunk)
    pprint(sdom)