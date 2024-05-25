import unittest

from shrimpgrad import Tensor
from shrimpgrad.engine.schedule import Scheduler
from shrimpgrad.runtime.ops import LoadOps 

class ScheduleTest(unittest.TestCase):
  def test_schedule_basic(self):
    x = Tensor.full((2,2), 2.0)

    s = Scheduler([x.thunk])
    schedule = s.schedule()

    self.assertEqual(1, len(schedule))
    self.assertEqual(schedule[0].ast.op, LoadOps.COPY)
 