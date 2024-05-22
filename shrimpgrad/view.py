from __future__ import annotations
from itertools import accumulate
import operator
from typing import Tuple
from shrimpgrad.device import Device
from shrimpgrad.dtype import DType
from shrimpgrad.util import prod

class View:
  """A description of how a tensors data is interpreted
  """
  def __init__(self, device: Device, shape: Tuple[int,...], dtype: DType):
    self.device, self.shape, self.dtype = device, shape, dtype
    self.ndim = len(self.shape)
    self.numel = prod(self.shape)
    self.elemsize = dtype.bytes
    self.nbytes = dtype.bytes * self.numel
    self._strides = self.strides()
    self._contiguous = self.contiguous() 
    self._scalar = self.ndim == 0
  def strides(self) -> Tuple[int,...]:
    return list(accumulate(self.shape[-1:0:-1], func=operator.mul, initial=(1 if len(self.shape)else None)))[::-1]
  def contiguous(self) -> bool:
    return all(self._strides[i] == self.shape[i+1]*self._strides[i+1] for i in range(0, self.ndim-1))
  def scalar(self): return self._scalar

  def reshape(self, new_shape: Tuple[int,...]) -> View:
    assert prod(new_shape) == self.numel, f'shape \'{new_shape}\' is invalid for input of size {self.numel}'
    return View(self.device, new_shape, self.dtype)  

  def permute(self, order: Tuple[int,...]) -> View:
    new_shape = [self.shape[i] for i in order]
    new_strides = [self._strides[i] for i in order]
    out = View(self.device, tuple(new_shape),  dtype=self.dtype) 
    out._strides = new_strides
    out._contiguous = False 
    return out
  
  def expand(self, shape: Tuple[int,...]) -> View:
    out = View.from_view(self) 
    for i, (si, so) in enumerate(zip(self.shape, shape)):
      if si != so: out._strides[i] = 0
    out.shape = shape
    return out
  
  def cast(self, dtype: DType) -> View:
    return View(self.device, self.shape, dtype)

  @staticmethod
  def from_view(view: View):
    return View(view.device, view.shape, view.dtype)