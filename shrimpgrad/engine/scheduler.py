
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Union, Sequence
from shrimpgrad.device import ConstBuffer, MemBuffer
from shrimpgrad.future import IndexedForwardGraph, Thunk
from shrimpgrad.engine.fuse_ops import FusionEngine
from shrimpgrad.runtime.ops import LoadOps, Op

@dataclass
class MidIR:
  ops: Sequence[Op]
  ins: Sequence[Sequence[Union[MemBuffer, ConstBuffer]]]
  out: Sequence[Union[MemBuffer, ConstBuffer]]
  args: List[Any]

@dataclass
class FusedKernel:
  computation: MidIR

class FusedKernelBuilder:
  def __init__(self, out: Thunk):
    self.g = IndexedForwardGraph(out)

    fusion_engine = FusionEngine(self.g)
    self.fused_ops = fusion_engine.fuse()
    self.groups = fusion_engine.groups

  def schedule_fused(self) -> List[FusedKernel]:
    kernels = []
    for group in self.groups[::-1]:
      if group.parent is None:
        if group.root not in self.fused_ops:
          thunk = group.root
          inputs = thunk.get_input_buffers()
          output = thunk.get_output_buffer()
          if thunk.realized is not None and thunk._op is not LoadOps.ASSIGN: continue
          ir = MidIR([thunk._op], [inputs], [output], [thunk.arg])
          kernels.append(FusedKernel(ir))
        else:
          fused = list(reversed(self.fused_ops[group.root])) + [group.root]
          inputs = [t.get_input_buffers() for t in fused]
          # Get the output for the last thunk
          output = [t.get_output_buffer() for t in fused]
          ops = [t._op for t in fused]
          args = [t.arg for t in fused]
          kernels.append(FusedKernel(MidIR(ops, inputs, output, args)))
    return kernels

  def schedule(self) -> List[FusedKernel]:
    return self.schedule_fused()

def print_schedule(schedule: List[FusedKernel]) -> None:
  print(f"SCHEDULE length={len(schedule)}")
  print("---------------------------------------------")
  for i, k in enumerate(schedule):
    ir = k.computation
    print(f"kernel {i+1} with {len(ir.ops)} steps:")
    print("-------------------------------------------")
    print(f" operation={ir.ops}")
    print("   inputs:")
    print_inputs(ir.ins)
    print("   outputs:")
    print_inputs([ir.out])
    print(f"   args={ir.args}")
    print("------------------------------------------")

def print_inputs(ins):
  for i in ins:
    for ii in i:
      if isinstance(ii, MemBuffer):
        print(f"    buffer {id(ii.buff)} with size {ii.buff.nbytes} bytes")
      if isinstance(ii, ConstBuffer):
        print(f"    constant {ii.value}")

