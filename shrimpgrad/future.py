from __future__ import annotations
from typing import List, Optional, Tuple, Union
from shrimpgrad.device import CPU, Device, Buffer
from shrimpgrad.dtype import ConstType, DType 
from shrimpgrad.runtime.ops import BinaryOps, LoadOps, Op, ReduceOps, TernaryOps, UnaryOps
from shrimpgrad.view import View

# Thunk
#   - shapetracker
#      - list of views (each view has shape, strides, contiguous(are the strides swapped?), offset?, mask (for padding)?)
#      - the shapetracker consolidates all the movement ops on a thunk
#      - some thunks will have an st with multiple views (ex. permute(1,0).reshape((..)) b/c the permute makes the view non-contiguous
#        just reshaping after will lose the effect the permute had on the tensor
#   - base_thunk (the thunk that has the real buffer with data)
#     - a newly loaded thunk will own the base buffer
#     - movement ops merge with the st.views[-1] view if possible and return a new shapetracker which in turn 
#       creates a new thunk (which is a view b/c it's self.base != self)
class Thunk:
  """A lazy evaluated buffer described by future computation or memory operation.
  Thunks are realized only when they are needed for an operation. Thunks form an abstract
  syntax tree where root thunks have allocated buffers (inputs usually from tensor factory methods)
  and child thunks are described by operations and their parent operands.
  """
  def __init__(self, device: Device, dtype: DType, view: View, operands: Tuple[Thunk, ...], op: Optional[Op]=None, data: Union[ConstType, List, bytes, memoryview]=None, base: Optional[Thunk]=None, arg=None):
    # initial buffer conditions
    self._view, self.device, self.dtype, self.arg = view, device, dtype, arg
    self._op, self._operands = op, operands 
    
    # Am I the base thunk? Does the buffer reside with me or another thunk? Am I a view of the base thunk after a movement op? (same questions)
    # The base has no base because it is the base (so based)
    self._base = None
    if base is None:
      # I'm the base I own the real buffer 
      self.buff = Buffer(self.device, self._view.numel, self.dtype)
    else:
      assert base.base == base, "base must be the base"
      self._base = base

  @property
  def shape(self): return self._view.shape
  @property
  def numel(self): return self._view.numel
  @property
  def scalar(self): return self._view.scalar
  @property
  def ndim(self): return self._view.ndim
  @property
  def base(self): return self._base if self._base is not None else self
  @property
  def realized(self) -> Optional[Buffer]: return self.buff if hasattr(self, 'buff') and self.buff.allocated else None
  
  @staticmethod
  def from_compute(op: Union[BinaryOps, UnaryOps, TernaryOps, ReduceOps], operands: Tuple[Thunk,...], view: View, device: Device, dtype: DType):
    if op in BinaryOps: assert len(operands) > 1, f'binary ops require two operands, {len(operands)} given' 
    if op in UnaryOps: assert len(operands) > 0, f'unary ops require one operands, {len(operands)} given'
    if op in TernaryOps: assert len(operands) > 2, f'ternary ops require three operands, {len(operands)} given' 
    return Thunk(device, dtype, View.from_view(view), tuple(operands), op)

  def alu(self, op: Union[UnaryOps, BinaryOps, TernaryOps], *in_thunks: Tuple[Thunk,...]) -> Thunk:
    return Thunk.from_compute(op, (self, *in_thunks), self._view, self.device, self.dtype)

  def reduce(self, op: ReduceOps, axis: Tuple[int,...]) -> Thunk: 
    new_shape = tuple([1 if i in axis else s for i,s in enumerate(self._view.shape)])
    return Thunk(self.device, self.dtype, View(new_shape), (self,), op, arg=axis)

  def reshape(self, shape: Tuple[int,...]):
    return Thunk(self.device, self.dtype, self._view.reshape(shape), (), base=self.base)

  def permute(self, order:Tuple[int,...]) -> Thunk:
    return Thunk(self.device, self.dtype, self._view.permute(order), (), base=self.base)

  def expand(self, shape: Tuple[int,...]) -> Thunk:
    return Thunk(self.device, self.dtype, self._view.expand(shape), (), base=self.base)

  def cast(self, dtype: DType) -> Thunk:
    return Thunk(self.device, dtype, self._view, (self,))

  @staticmethod 
  def load_from_cpu(data, dtype, shape):
    thunk = Thunk.loadop(LoadOps.EMPTY, shape, dtype, CPU())
    thunk.buff.allocate(with_data=data)
    # This ensures we schedule it early for realize
    del thunk._operands
    return thunk
  
  @staticmethod
  def loadop(op: LoadOps, shape, dtype, device, arg=None, srcs=()):
    return Thunk(device, dtype, View(shape), srcs, op=op, arg=arg)
  
  def copy_to_device(self, device: Device) -> Thunk:
    # Generaly self is a LoadOps.EMPTY with device as CPU
    # It may have been reshaped etc so ensure we copy from the base
    return Thunk(device, self.dtype, self._view, (self.base, ), LoadOps.COPY, arg=self.base.buff.nbytes)

  def const(self, val: ConstType, shape: Tuple[int,...]=None):
    shape = self.shape if shape is None else shape
    return Thunk.loadop(LoadOps.CONST, tuple(), self.dtype, self.device, arg=val).reshape((1,)*len(shape)).expand(shape)

  def __repr__(self) -> str:
    return f"<THUNK {self.device} {self.shape} {str(self.dtype)[7:]} {self._op}>"
  def __hash__(self): return id(self)