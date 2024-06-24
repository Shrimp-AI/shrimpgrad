# Given a schedule of FusedKernels
#   - estimate worst case number of FLOPS
#   - estimate worst case memory

from typing import List

from shrimpgrad.device import MemBuffer
from shrimpgrad.engine.scheduler import FusedKernel
from shrimpgrad.runtime.ops import BinaryOps, ReduceOps, UnaryOps
from shrimpgrad.util import prod
from shrimpgrad.engine.scheduler import print_schedule

def flop_counter(schedule: List[FusedKernel]) -> int:
  # Either a fused kernel (multiple ops)
  # or a single op kernel
  flop = 0
  for fk in schedule:
    c = fk.computation
    for op, inp in zip(c.ops, c.ins):
      if op  in BinaryOps or op in UnaryOps or op in ReduceOps:
        flop += prod(inp[0].vt.shape)
  return flop

def memory_estimator(schedule: List[FusedKernel]) -> int:
  mem = 0
  print_schedule(schedule)
  for fk in schedule:
    c = fk.computation
    for inp, out in zip(c.ins, c.out):
      # all inputs are loaded
      # all outputs are stored
      for i in inp:
        if i.__class__ == MemBuffer:
          mem += i.buff.nbytes
      if out.__class__ == MemBuffer:
        mem += out.buff.nbytes

  return mem