from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union
from shrimpgrad.dtype import ConstType, DType
from shrimpgrad.runtime.ops import BinaryOps, BufferOps, LoadOps, Op, ReduceOps, TernaryOps, UnaryOps
from shrimpgrad.tensor import Tensor 

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

  def resolve(self) -> Tensor: 
    pass
