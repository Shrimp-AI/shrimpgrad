import ctypes
from typing import Callable, List
import unittest

from shrimpgrad.device import MemBuffer
from shrimpgrad.engine.runner import BufferCopy, _gen_load_kernels
from shrimpgrad import Tensor, nn
from shrimpgrad.engine.scheduler import FusedKernelBuilder, print_schedule
from shrimpgrad.runtime.ops import LoadOps

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
    copy_kernel = BufferCopy(dst, src, size)
    copy_kernel()
    for i, val in enumerate(dst.buff._pointer(ctypes.c_float)[0:4]):
      assert val == data[i]

  def test_two_copy(self):
    x = Tensor((2,2), data0:=[1.0, 2.0, 3.0, 4.0])
    y = Tensor((2,2), data1:=[5.0, 6.0, 7.0, 8.0])
    out = x + y
    fkb = FusedKernelBuilder(out.thunk)
    schedule = fkb.schedule()
    print_schedule(schedule)

    for i, fk in enumerate(schedule):
      if len(fk.computation.ins) == 1 and fk.computation.ops[0] == LoadOps.COPY:
        src = fk.computation.ins[0][0]
        dst = fk.computation.out[0]
        size = fk.computation.args[0]

        assert isinstance(src, MemBuffer)
        assert isinstance(dst, MemBuffer)
        copy_kernel = BufferCopy(dst, src, size)
        copy_kernel()
        for j, val in enumerate(dst.buff._pointer(ctypes.c_float)[0:4]):
          if i == 0:
            assert val == data0[j]
          else:
            assert val == data1[j]


  def test_shallow_net(self):
    class Model:
      def __init__(self):
        self.layers: List[Callable[[Tensor], Tensor]] = [
        nn.Linear(2, 2), Tensor.relu,
        nn.Linear(2, 2), Tensor.relu,
        nn.Linear(2, 2), Tensor.relu,
        nn.Linear(2, 1), Tensor.sigmoid,
        ]
      def __call__(self, x: Tensor):
        return x.sequential(self.layers)

    x = Tensor.randn(2,2)
    model = Model()
    out = model(x)
    out.realize()
    print(out.thunk.buff.pointer(ctypes.c_float)[0:2])
    out.backward()

  def test_simple(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    z = x+y
    z.realize()
    print(z.numpy())  
    
  def test_load_const(self):
    x = Tensor.full((2,2,2), 3.0)
    fkb = FusedKernelBuilder(x.thunk)
    schedule = fkb.schedule()
    buff_copy, kernels = _gen_load_kernels(schedule)
    self.assertEqual([], buff_copy)
    self.assertEqual(0, len(kernels))