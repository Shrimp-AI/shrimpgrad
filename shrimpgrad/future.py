from __future__ import annotations
import struct
from typing import List, Optional, Tuple, Union
from shrimpgrad.dtype import ConstType, DType, dtypes
from shrimpgrad.runtime.ops import BinaryOps, BufferOps, LoadOps, Op, ReduceOps, TernaryOps, UnaryOps, MovementOps
from shrimpgrad.view import View
from shrimpgrad.memory.buffer import Buffer

class Thunk:
  """A lazy evaluated buffer described by future computation or memory operation.
  Thunks are realized only when they are needed for an operation. Thunks form an abstract
  syntax tree where root thunks have allocated buffers (inputs usually from tensor factory methods)
  and child thunks are described by operations and their parent operands.
  """
  def __init__(self, op: Optional[Op], operands: Tuple[Thunk, ...], view: View, data: Union[ConstType, List, bytes, memoryview]=None):
    # initial tensor conditions
    self._view = view
    self._op, self._operands = op, operands 
    # Unallocated buffer
    self.buff = Buffer(self._view.device._allocator(), self._view.numel, self._view.dtype)
    if data is not None and isinstance(data, List):
      # Could be a root thunk
      # We have data, need to allocate and load the buffer 
      # TODO: Support other data types and dtypes (look at datas type in function sig)
      if self._view.dtype == dtypes.float32:
        self.buff.allocate()
        self.buff.copyin(struct.pack('f'*len(data), *data))
  
  @staticmethod
  def from_compute(op: Union[BinaryOps, UnaryOps, TernaryOps, ReduceOps], operands: Tuple[Thunk,...], view: View):
    if op in BinaryOps: assert len(operands) > 1, f'binary ops require two operands, {len(operands)} given' 
    if op in UnaryOps: assert len(operands) > 0, f'unary ops require one operands, {len(operands)} given'
    if op in TernaryOps: assert len(operands) > 2, f'ternary ops require three operands, {len(operands)} given' 
    return Thunk(op, tuple(operands), View.from_view(view))
    
  @staticmethod
  def from_memory(op: Union[LoadOps, BufferOps], view: View, data: Union[memoryview, ConstType]) -> Thunk:
    if op == LoadOps.EMPTY: return Thunk(op, (),  view)
    if op == LoadOps.CONTIGUOUS: assert isinstance(data, memoryview), 'data for contiguous loads must come from a memoryview'
    if op == LoadOps.CONST: assert isinstance(data, ConstType), 'data for const loads must come from a ConstType'
    return Thunk(op, (), view,  data=data) 
    
  def load(self, op: LoadOps, data: Union[ConstType, memoryview]) -> Thunk:  return Thunk.from_memory(op, self._view, data)
    
  def alu(self, op: Union[UnaryOps, BinaryOps, TernaryOps], *in_thunks: Tuple[Thunk,...]) -> Thunk:
    return Thunk.from_compute(op, (self, *in_thunks), self._view)

  def reduce(self, op: ReduceOps, axis: Tuple[int,...]) -> Thunk: 
    new_shape = tuple([1 if i in axis else s for i,s in enumerate(self._view.shape)])
    return Thunk(op, (self,), View(self._view.device, new_shape, self._view.dtype))

  def reshape(self, shape: Tuple[int,...]):
    return Thunk(MovementOps.RESHAPE, (self,), self._view.reshape(shape))

  def permute(self, order:Tuple[int,...]) -> Thunk:
    return Thunk(MovementOps.PERMUTE, (self,), self._view.permute(order))

  def expand(self, shape: Tuple[int,...]) -> Thunk:
    return Thunk(MovementOps.PERMUTE, (self,), self._view.expand(shape))

  def cast(self, dtype: DType) -> Thunk:
    return Thunk(MovementOps.CAST, (self,), self._view.cast(dtype))

  def const(self, val: ConstType):
    return Thunk(LoadOps.CONST, (self,), self._view, val)

  def __repr__(self) -> str:
    return f"<THUNK {self._view.device} {self._view.shape} {str(self._view.dtype)[7:]} {self._op}>"
  
def _tree(thunk: Thunk, prefix="") -> str:
  if len(thunk._operands) == 0: return [f"━━ {prefix}{thunk._op.name}"]
  lines = [f"━┳ {prefix}{thunk._op.name}"]
  childs = [_tree(c) for c in thunk._operands[:]]
  for c in childs[:-1]: lines += [f" ┣{c[0]}"] + [f" ┃{l}" for l in c[1:]]
  return lines + [" ┗"+childs[-1][0]] + ["  "+l for l in childs[-1][1:]]

def print_ast(thunk: Thunk): print("\n".join([f"{str(i).rjust(3)} {s}" for i,s in enumerate(_tree(thunk))]))
