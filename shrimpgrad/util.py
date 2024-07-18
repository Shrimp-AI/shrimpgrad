import contextlib
from functools import reduce
import time
from typing import Any, Iterable, Tuple, TypeVar
import operator
import math
from typing_extensions import Protocol

def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def prod(x: Iterable[int]) -> int: return reduce(operator.mul, x, 1)
def dedup(x: Iterable[Any]) -> Iterable[Any]: return list(set(x))  

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

class Timing(contextlib.ContextDecorator):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, *exc):
    self.et = time.perf_counter_ns() - self.st
    if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

# Typing helper for ensuring an object accepts __get_item__
K = TypeVar('K', contravariant=True)
V = TypeVar('V', covariant=True)
class SupportsGetItem(Protocol[K, V]):
  def __getitem__(self, __key: K) -> V: ...
