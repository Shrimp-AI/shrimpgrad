import ctypes
import unittest

from shrimpgrad.device import MemBuffer
from shrimpgrad.engine.runner import BufferCopy
from shrimpgrad import Tensor
from shrimpgrad.engine.scheduler import FusedKernelBuilder, print_schedule

class TestRunner(unittest.TestCase):
  # TODO: Test later when we test all of python device
  def test_buffer_copy(self):
    x = Tensor((2,2), data:=[1.0, 2.0, 3.0, 4.0])
    fkb = FusedKernelBuilder(x.thunk)
    schedule = fkb.schedule()
    print_schedule(schedule)
    fk = schedule[0]
    src = fk.computation.ins[0][0]
    dst = fk.computation.out[0]
    size = fk.computation.args[0]

    assert isinstance(src, MemBuffer)
    assert isinstance(dst, MemBuffer)
    copy_kernel = BufferCopy(dst.buff, src.buff, size)
    copy_kernel()
    for i, val in enumerate(dst.buff.pointer(ctypes.c_float)[0:4]):
      assert val == data[i]



