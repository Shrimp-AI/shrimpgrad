from functools import reduce
from typing import Iterable, Optional, Tuple, Union
import operator
import math

def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def prod(x: Iterable[int|float]) -> Union[float, int]: return reduce(operator.mul, x, 1)

def calc_loops(tensor, key: Optional[slice|int]) -> Iterable[int]:
  if not key and not isinstance(key, int):  key = [slice(0, dim, 1) for dim in tensor.view.shape]
  if isinstance(key, int) or isinstance(key, slice): key = (key,)
  if len(key) > len(tensor.view.shape): raise IndexError(f'index of {key} is larger than dim {tensor.view.shape}.')
  extra_dim = 0
  if len(key) < len(tensor.view.shape): extra_dim = len(tensor.view.shape) - len(key)
  loops = []
  for i, k in enumerate(key):
    if isinstance(k, int):
      if k >= tensor.view.shape[i]:  raise IndexError(f'index of {k} is out of bounds for dim with size {tensor.view.shape[i]}')
      start = k
      if start < 0:
        if abs(start) > tensor.view.shape[i]:
          raise IndexError(f'index of {start} is out of bounds for dim with size {tensor.view.shape[i]}')
        k = tensor.view.shape[i] + start
      start, end = k, k + 1
      loops.append((start,end, 1))
    elif isinstance(k, slice):
      start, end, step = k.indices(tensor.view.shape[i])
      if start < 0:
        if abs(start) > tensor.view.shape[i]:
          raise IndexError(f'index of {start} is out of bounds for dim with size {tensor.view.shape[i]}')
        start = tensor.view.shape[i] + start 
      if end < 0:
        if abs(end) > tensor.view.shape[i]:
          raise IndexError(f'index of {end} is out of bounds for dim with size {tensor.view.shape[i]}')
        end = tensor.view.shape[i] + end
      if start >= end: return [] 
      loops.append((start, end, step))
  if extra_dim: 
    for ed in range(len(key), len(key) + extra_dim): loops.append((0, tensor.view.shape[ed], 1))
  return loops

def to_nested_list(tensor, key: Optional[slice|int]) -> Iterable:
  def build(dim:int, offset:int, loops:Iterable[tuple], result:Iterable[float|int]) -> Iterable[float|int]:
    if not loops: return result 
    s, e, step = loops[0]
    for i in range(s, e, step):
      if len(loops) == 1: 
        result.append(tensor.data[offset+i*step*tensor.view.strides()[dim]])
      else: 
        result.append([])
        build(dim+1, offset + i*tensor.view.strides()[dim]*step, loops[1:], result[-1])
    return result 
  return build(0, 0, calc_loops(tensor, key), []) 

def flatten(tensor):
  def build(dim:int, offset:int, loops:Iterable[tuple], result:Iterable[float|int]) -> Iterable[float|int]:
    if not loops: return result 
    s, e, step = loops[0]
    for i in range(s, e, step):
      if len(loops) == 1: 
        result.append(tensor.data[offset+i*step*tensor.view.strides()[dim]])
      else: 
        build(dim+1, offset + i*tensor.view.strides()[dim]*step, loops[1:], result)
    return result 
  return build(0, 0, calc_loops(tensor, None), [])  

## Used for Kaiming init
def calc_fan_in_fan_out(shape:Tuple[int,...]):
  # Similar to pytorch 
  dim = len(shape)
  assert dim >= 2, 'fan in/out requires tensors with dim >= 2'
  r = prod(shape[2:]) if dim > 2 else 1
  return shape[1] * r, shape[0] * r 

def calc_gain(a=0.01):
  # TODO: Only support leaky ReLU which is used for affine transformation
  # in torch
  return math.sqrt(2.0 / (1.0 + a ** 2.0))