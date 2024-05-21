from __future__ import annotations
from typing import Any, List, Tuple, Union
from shrimpgrad.dtype import ConstType, DType
from shrimpgrad.runtime.ops import BinaryOps, BufferOps, LoadOps, Op, ReduceOps, TernaryOps, UnaryOps
from shrimpgrad.tensor import Tensor
from shrimpgrad.memory.buffer import Buffer, Allocator

def create_dependent_future_tensor(): pass
def create_independent_future_tensor(): pass

class FutureTensor:
  """An unresolved tensor described by future computations. 
  """
  def __init__(self, op: Op, in_tensors: Tuple[FutureTensor, ...], shape: Tuple[int,...], dtype: DType, device:str):
    # initial tensor conditions
    self._shape, self._dtype, self._device = shape, dtype, device
    self._op, self._in_tensors = op, in_tensors
    
  def load(self, op: LoadOps, src: Union[ConstType, List[Any]]) -> FutureTensor: 
    pass
    
  def alu(self, op: Union[UnaryOps, BinaryOps, TernaryOps], *in_tensors: Tuple[FutureTensor,...]) -> FutureTensor:
    return FutureTensor(op, in_tensors, self._shape, self._dtype, self._device)

  def reduce(self, op: ReduceOps, in_tensors: Tuple[FutureTensor, ...], *args, **kwargs) -> FutureTensor:
    pass

  def buffer(self, op: BufferOps, in_tensors: Tuple[FutureTensor], *args, **kwargs) -> FutureTensor:
    pass
  
  def const(self, val:ConstType) -> FutureTensor:
    pass
  
  def reshape(self, shape: Tuple[int,...]):
    # if prod(shape) != x.numel: raise RuntimeError(f'shape \'{shape}\' is invalid for input of size {x.numel}')
    # if x.contiguous:
    #   if shape == x.shape:
    #     return FutureTensor(shape, x.data, dtype=x.dtype)
    #   # Reshape from scalar to n-dim tensor 
    #   if x.is_scalar() and len(shape):
    #     return FutureTensor(shape, [x.data], dtype=x.dtype)
    #   return FutureTensor(shape, x.data if len(shape) else x.data[0], dtype=x.dtype)
    # return FutureTensor(shape, flatten(x), dtype=x.dtype)
    pass

  def permute(self, order:Tuple[int,...]) -> FutureTensor:
    # new_shape = [x._shape[i] for i in order]
    # new_strides = [x._strides[i] for i in order]
    # out = FutureTensor(tuple(new_shape), x._data, dtype=x._dtype) 
    # out.strides = new_strides
    # out.contiguous = False
    pass

  def expand(self, shape: Tuple[int,...]) -> FutureTensor:
    # out = FutureTensor.zeros_like(x)
    # for i, (si, so) in enumerate(zip(x.shape, shape)):
    #   if si != so: 
    #     out.strides[i] = 0
    #     ctx.expanded_axis.append(i)
    # out.shape = shape
    # out.data = x.data
    pass

  def cast(self, dtype: DType) -> FutureTensor:
    #  x.data = list(map(functools.partial(dtypes.cast, dtype), x.data)) if not x.is_scalar() else dtypes.cast(dtype, x.data)
    pass

  def resolve(self) -> Tensor: 
    pass
