from __future__ import annotations
from itertools import accumulate
import operator
from typing import Tuple
from shrimpgrad.util import prod

class View:
  """A description of how a thunk's data is interpreted
  """
  def __init__(self, shape: Tuple[int,...]):
    self.shape = shape 
    self._strides = tuple(accumulate(self.shape[-1:0:-1], func=operator.mul, initial=(1 if len(self.shape)else None)))[::-1]

  @property
  def strides(self) -> Tuple[int,...]: return self._strides

  @property
  def contiguous(self) -> bool:
    return all(self._strides[i] == self.shape[i+1]*self._strides[i+1] for i in range(0, self.ndim-1))

  @property
  def scalar(self): return self.ndim == 0 
  @property 
  def numel(self): return prod(self.shape)    
  @property
  def ndim(self): return len(self.shape)

  def reshape(self, new_shape: Tuple[int,...]) -> View:
    if len(self.shape):
      assert prod(new_shape) == self.numel, f'shape \'{new_shape}\' is invalid for input of size {self.numel} of shape {self.shape}'
      return View(new_shape)  
    return View(new_shape)  

  def permute(self, order: Tuple[int,...]) -> View:
    new_shape = tuple([self.shape[i] for i in order])
    new_strides = tuple([self.strides[i] for i in order])
    v = View(new_shape) 
    v._strides = new_strides
    return v
    
  def expand(self, shape: Tuple[int,...]) -> View:
    out = View.from_view(self) 
    strd = list(out.strides)
    for i, (si, so) in enumerate(zip(self.shape, shape)):
      if si != so: strd[i] = 0
    out.shape = shape
    out._strides = strd
    return out
  
  @staticmethod
  def from_view(view: View):
    return View(view.shape)