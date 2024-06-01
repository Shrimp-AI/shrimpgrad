
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Union, Sequence
from shrimpgrad.device import ConstBuffer, MemBuffer
from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.future import IndexedForwardGraph, Thunk
from shrimpgrad.engine.fuse_ops import FusionEngine
from shrimpgrad.runtime.ops import Op

@dataclass
class MidIR:
  op: Op
  ins: Sequence[Union[MemBuffer, ConstBuffer]]
  out: Union[MemBuffer, ConstBuffer]
  arg: Any

@dataclass
class FusedKernel:
  computation: List[MidIR]

class FusedKernelBuilder:
  def __init__(self, out: Thunk):
    log_thunk(out)
    self.g = IndexedForwardGraph(out)

    fusion_engine = FusionEngine(self.g)
    self.fused_ops = fusion_engine.fuse()
    self.groups = fusion_engine.groups

  def schedule_fused(self): 
    kernels = []
    for group in self.groups[::-1]:
      if group.parent is None:
        if group.root not in self.fused_ops: 
          thunk = group.root
          inputs = thunk.get_input_buffers()
          output = thunk.get_output_buffer()
          ir = MidIR(thunk._op, inputs, output, arg=thunk.arg)
          kernels.append(FusedKernel([ir])) 
        else:
          fused = list(reversed(self.fused_ops[group.root])) + [group.root]
          fused_kernel = []
          for thunk in fused:
            inputs = thunk.get_input_buffers()
            output = thunk.get_output_buffer()
            ir = MidIR(thunk._op, inputs, output, arg=thunk.arg)
            fused_kernel.append(ir)
          kernels.append(FusedKernel(fused_kernel))
    return kernels  

           
  def schedule(self):
    return self.schedule_fused()

def print_schedule(schedule: List[FusedKernel]) -> None:
  print(f"SCHEDULE length={len(schedule)}")
  print("---------------------------------------------")
  for i, k in enumerate(schedule):
    print(f"kernel {i+1} with {len(k.computation)} steps:")
    print("-------------------------------------------")
    for i, comp in enumerate(k.computation): 
      print(f"step {i}:")
      print(f" operation={comp.op}")
      print("   inputs:")
      print_inputs(comp.ins)
      print("   outputs:")
      print_inputs([comp.out])
      print(f"   arg={comp.arg}")
    print("------------------------------------------")
    
def print_inputs(ins):
  for i in ins:
    if isinstance(i, MemBuffer):
      print(f"    buffer {id(i.buff)} with size {i.buff.nbytes} bytes")
    if isinstance(i, ConstBuffer):
      print(f"    constant {i.value}")

