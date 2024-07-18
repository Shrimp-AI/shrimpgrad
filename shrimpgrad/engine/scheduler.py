
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Sequence
from shrimpgrad.device import ConstBuffer, MemBuffer
from shrimpgrad.future import IndexedForwardGraph, Thunk
from shrimpgrad.engine.fuse_ops import FusionEngine
from shrimpgrad.runtime.ops import LoadOps, Op

@dataclass
class MidIR:
  ops: Sequence[Op]
  ins: Sequence[Sequence[MemBuffer]]
  out: Sequence[MemBuffer]
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
    visited = set()
    for group in self.groups[::-1]:
      if group.parent is None:
        if group.root not in self.fused_ops:
          thunk = group.root
          if thunk in visited: continue
          visited.add(thunk)
          inputs = thunk.get_input_buffers()
          output = thunk.get_output_buffer()
          # Skip copies that are realized so we don't overwrite forward pass values.
          # This is crucial for backwards passes functioning properly.
          # Assign thunks can be realized in forward passes as their backing buffer
          # is realized. This is useful for optimizer updates to params.
          is_realized = thunk.realized is not None
          if is_realized and thunk._op is not LoadOps.ASSIGN: continue 
          if thunk._op is LoadOps.ASSIGN and thunk._operands[1].realized is not None : 
            assert  is_realized, 'assign target must be realized'
            continue  
          assert thunk._op is not None, 'no views should be here'
          ir = MidIR([thunk._op], [inputs], [output], [thunk.arg])
          kernels.append(FusedKernel(ir))
        else:
          fused = list(reversed(self.fused_ops[group.root])) + [group.root]
          fused_unrealized = []
          # If a thunk in this fusion group is already realized, there's no
          # need to schedule it again, it's backing buffer already has
          # the result that we would compute again causing a doubling of the buffer .
          # This is vital when running
          # backward passes because backward functions sometimes use input thunks that
          # become realized in the forward pass. When backward includes those thunks
          # in the backwards graph, it will cause those forward sub-graphs to be re-executed
          # here we eliminate this duplicate computation by not scheduling thunks that are
          # realized. In a forward pass the only realized thunks are LoadOps.EMPTY, but they
          # don't make it to the scheduling phase. The only time we skip here is in backwards
          # graph scheduling.
          for thunk in fused:
            if thunk in visited: continue
            visited.add(thunk)
            is_realized = thunk.realized is not None
            if is_realized and thunk._op is not LoadOps.ASSIGN: continue
            if thunk._op is LoadOps.ASSIGN and thunk._operands[1].realized is not None : 
              assert  is_realized, 'assign target must be realized'
              continue
            fused_unrealized.append(thunk)
          if not fused_unrealized: continue
          fused = fused_unrealized
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

