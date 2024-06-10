import ctypes
import unittest

from shrimpgrad.device import MemBuffer
from shrimpgrad.engine.runner import BufferCopy
from shrimpgrad import Tensor
from shrimpgrad.engine.scheduler import FusedKernelBuilder
from shrimpgrad.runtime.ops import LoadOps

class TestRunner(unittest.TestCase):
  # TODO: Test later when we test all of python device
  def test_buffer_copy(self):
    x = Tensor.rand(2,2)
    y = Tensor.rand(2,2)
    out = x + y
    fkb = FusedKernelBuilder(out.thunk)
    schedule = fkb.schedule()
    for fk in schedule:
      if len(fk.computation.ins) == 1 and len(fk.computation.out) == 1 and fk.computation.ops[0] is LoadOps.COPY:
        print(fk)
        src = fk.computation.ins[0][0]
        dst = fk.computation.out[0]
        size = fk.computation.args[0]
        print(f"{size = }")

        assert isinstance(src, MemBuffer)
        assert isinstance(dst, MemBuffer)
        copy_kernel = BufferCopy(dst.buff, src.buff, size)
        copy_kernel()
        for val in dst.buff.pointer(ctypes.c_float)[0:4]:
            print(val)



