from functools import reduce
from typing import Iterable, Optional, Tuple, Union
import operator
import math

def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def prod(x: Iterable[int]) -> int: return reduce(operator.mul, x, 1)

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

# A forward topological sort of the tensor
# graph. Basically just a Post-DFS traversal.
def deepwalk(x):
  visited = set() 
  def walk(x):
    if x in visited:
      return
    visited.add(x)
    if not x.ctx:
      yield x 
      return
    for t in x.ctx.tensors:
      yield from walk(t)
    yield x 
  yield from walk(x)

def dump_tensors(x):
  for t in deepwalk(x):
    t.analyze()