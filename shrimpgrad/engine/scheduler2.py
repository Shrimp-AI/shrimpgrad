
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Union
from shrimpgrad.device import Buffer, Device
from shrimpgrad.dtype import ConstType
from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.future import IndexedForwardGraph, Thunk
from shrimpgrad.engine.fuse_ops import FusionEngine
from shrimpgrad.runtime.ops import LoadOps, Op
from shrimpgrad.view import View


def const_folding():
  pass


def schedule_loads():
  pass


def gen_fused_kernels():
  pass


def schedule_fused_kernels():
  pass

@dataclass
class MemBuffer:
  buff: Buffer
  view: View

@dataclass
class ConstBuffer:
  value: ConstType
  device: Device

@dataclass
class MidIR:
  op: Op
  ins: List[Union[MemBuffer, ConstBuffer]]
  outs: List[Union[MemBuffer, ConstBuffer]]
  arg: Any
  input_for: List[Thunk] 

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

    self.scheduled_kernels = []
  
  def schedule_loads(self):
    for load, outs in self.loads.items():
      self.scheduled_kernels.append(self._schedule(load, outs))
  def schedule_consts(self):
    for const, outs in self.consts.items():
      self.scheduled_kernels.append(self._schedule(const, outs))
  
  def schedule_fused(self): 
    fused = []
    for out, chain in self.fused_ops.items():
      computations: List[Thunk] = chain + out
      for computation in computations:
        for operand in computation._operands:
          fused.append(self._schedule(operand, [computation]))
    return FusedKernel(fused)
          
  def _schedule(self, thunk: Thunk, input_for: Iterable[Thunk]):
    if thunk._op is LoadOps.COPY:
      ins = [MemBuffer(thunk._operands[0].buff, thunk._operands[0]._view)]
      arg = thunk.arg
      outs = [MemBuffer(thunk.buff, thunk._view)] 
      return MidIR(LoadOps.COPY, ins, outs, arg, input_for)
    if thunk._op is LoadOps.CONST:
      arg = thunk.arg
      return MidIR(LoadOps.CONST, [], [ConstBuffer(arg, thunk.device)], arg, input_for)


 
