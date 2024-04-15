from typing import Iterable, Self, TypeAlias, Union
from functools import reduce 
import operator
from pprint import pformat

Num: TypeAlias = Union[float, int, complex]
DType: TypeAlias = Union[float, int]
Shape: TypeAlias = tuple[int]

class IndexOutOfBounds(Exception):
  pass

def prod(xs: Iterable[int|float]) -> Union[float, int]:
  return reduce(operator.mul, xs, 1)

class Tensor:
  def __init__(self, shape: Shape, data: Iterable[Num]) -> Self:
    self.shape = shape 
    self.data = data
    self.size = prod(self.shape)
    self.strides = [] 
    out: int = 1
    for dim in self.shape: 
      out *= dim
      self.strides.append(self.size // out)
    key = []
    for dim in self.shape:
      s = slice(0, dim, 1)
      key.append(s)
    self.base_view = self.__build_view(tuple(key))

  def __build_view(self, key: Union[tuple, int]) -> Iterable:
    if isinstance(key, int) or isinstance(key, slice):
      key = (key,)
    if len(key) > len(self.shape):
      raise IndexOutOfBounds('Index out of bounds.')
    extra_dim = 0
    if len(key) < len(self.shape):
      extra_dim = len(self.shape) - len(key)
    loops = []
    for i, k in enumerate(key):
      if isinstance(k, int):
        if k >= self.shape[i]: 
          raise IndexOutOfBounds(f'Index out of bounds: {self.shape[i]} <= {k}')
        start = k
        if start < 0:
          start = self.shape[i] + start
        start, end = k, k + 1

        loops.append((start,end, 1))
      elif isinstance(k, slice):
        start, end, step = k.indices(self.shape[i])
        if start < 0:
          start = self.shape[i] + start 
        if end < 0:
          end = self.shape[i] + end
        if start >= end:
          return [] 
        loops.append((start, end, step))
    if extra_dim:
      start = len(key)
      for ed in range(start, start+extra_dim):
        loops.append((0, self.shape[ed], 1))

    def build(dim:int, offset:int, loops:Iterable[tuple], tensor:Iterable[Num]) -> Iterable[Num]:
      if not loops:
        return tensor
      s, e, step = loops[0]
      for i in range(s, e, step):
        if len(loops) == 1:
          # add elements
          offset += i*step
          tensor.append(self.data[offset])
          offset -= i*step
        else:
          # add dimension
          new_dim = []
          tensor.append(new_dim)
          offset += i*self.strides[dim]*step
          build(dim+1, offset, loops[1:], new_dim)
          offset -= i*self.strides[dim]*step
      return tensor 
    return build(0, 0, loops, []) 

  def __getitem__(self, key) -> Self:
    new_view = self.__build_view(key)
    new_tensor = Tensor(self.shape, self.data)
    new_tensor.base_view = new_view
    return new_tensor

  def __repr__(self):
    return f'tensor({pformat(self.base_view, width=40)})'

def zeros(shape: Shape) -> Tensor:
  return Tensor(shape, [0]*prod(shape))

def arange(x: int, shape: Shape) -> Tensor:
  return Tensor(shape, [i for i in range(x)]) 