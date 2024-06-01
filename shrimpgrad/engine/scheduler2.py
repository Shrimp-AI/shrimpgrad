
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Union, Sequence
from shrimpgrad.device import Buffer, ConstBuffer, Device, MemBuffer
from shrimpgrad.dtype import ConstType
from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.future import IndexedForwardGraph, Thunk
from shrimpgrad.engine.fuse_ops import FusionEngine
from shrimpgrad.runtime.ops import Op
from shrimpgrad.view import View

@dataclass
class MidIR:
  op: Op
  ins: Sequence[Union[MemBuffer, ConstBuffer]]
  outs: Union[MemBuffer, ConstBuffer]
  arg: Any

@dataclass
class FusedKernel:
  computation: List[MidIR]

class FusedKernelBuilder:
  def __init__(self, out: Thunk):
    log_thunk(out)
    self.g = IndexedForwardGraph(out)
    self.loads = self.g.saved[0]
    self.expands = self.g.saved[1]
    self.consts = self.g.saved[3]

    fusion_engine = FusionEngine(self.g)
    self.fused_ops, self.unfused = fusion_engine.fuse()
    self.groups = fusion_engine.groups

  def schedule_loads(self):
    loads = []
    for load in self.loads:
      inputs = load.get_input_buffers()
      output = load.get_output_buffer()
      ir = MidIR(load._op, inputs, output, arg=load.arg)
      loads.append(FusedKernel([ir]))
    return loads

  def schedule_consts(self):
    loads = []
    for load in self.consts:
      inputs = load.get_input_buffers()
      output = load.get_output_buffer()
      ir = MidIR(load._op, inputs, output, arg=load.arg)
      loads.append(FusedKernel([ir]))
    return loads
  
  def schedule_fused(self): 
    kernels = []
    for group in self.groups[::-1]:
      print(group)
      if group.parent is None:
        if group.root not in self.fused_ops: 
          thunk = group.root
          inputs = thunk.get_input_buffers()
          output = thunk.get_output_buffer()
          ir = MidIR(thunk._op, inputs, output, arg=thunk.arg)
          kernels.append(FusedKernel([ir])) 
        else:
          fused = self.fused_ops[group.root] + [group.root]
          fused_kernel = []
          for thunk in fused:
            inputs = thunk.get_input_buffers()
            output = thunk.get_output_buffer()
            ir = MidIR(thunk._op, inputs, output, arg=thunk.arg)
            fused_kernel.append(ir)
          kernels.append(FusedKernel(fused_kernel))
    return kernels  

           
  def schedule(self):
    kernels = []
    kernels.append(self.schedule_loads())
    kernels.append(self.schedule_consts())
    kernels.append(self.schedule_fused())
    return kernels


 
