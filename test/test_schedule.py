import unittest

from shrimpgrad import Tensor
from shrimpgrad.engine.schedule import Scheduler
from shrimpgrad.runtime.ops import BufferOps, LoadOps 

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