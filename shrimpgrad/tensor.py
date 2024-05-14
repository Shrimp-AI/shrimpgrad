from __future__ import annotations
from itertools import accumulate
import math
import operator
from typing import Iterable, List, Optional, Self, TypeAlias, Union, Tuple
from pprint import pformat
from shrimpgrad.dtype import DType, dtypes
from random import uniform, gauss
from shrimpgrad.util import calc_fan_in_fan_out, calc_gain, prod, to_nested_list 

Num: TypeAlias = Union[float, int, complex]
Shape: TypeAlias = Tuple[int, ...]

def pad_left(*shps: Tuple[int, ...], v=1) -> List[Tuple[int ,...]]: return [tuple((v,)*(max(len(s) for s in shps)-len(s)) + s) for s in shps]
def broadcast_shape(*shps: Tuple[int, ...]) -> Tuple[int, ...]: return tuple([max([s[dim] for s in shps]) for dim in range(len(shps[0]))])

class Tensor:
  def __init__(self, shape: Shape, data: Union[Iterable[Num], Num], dtype:DType=dtypes.float32, device='CPU', requires_grad:Optional[bool]=None) -> Self:
    self.shape, self.data, self.numel, self.strides, self.dtype = shape, data, prod(shape), [], dtype
    self.grad: Optional[Tensor] = None
    self.requires_grad = requires_grad 
    self.device = device
    self.ndim = len(self.shape)
    self.index_view = None # Set by get_item for use in repr
    # If this tensor is produced via a computation we
    # add this computational context enabling the backwards pass
    from shrimpgrad.autograd.function import Function
    self.ctx: Optional[Function] = None
    # Scalar value
    if self.is_scalar():
      # Ensure it's a number not dimensional data
      assert isinstance(data, Num)
      self.data = data
      self.contiguous = True
      return
    self._strides()
    self.contiguous = all(self.strides[i] == self.shape[i+1]*self.strides[i+1] for i in range(0, self.ndim-1))
  
  def is_scalar(self):
    return not self.ndim 

  def backward(self) -> Tensor:
    self.grad = Tensor.ones(self.shape, self.dtype)
    visited = set()
    topo = []
    # TODO: Turn this into generator so we don't allocate memory for 
    # a massive list
    def build_topo(tensor: Tensor) -> List[Tensor]:  
      if tensor not in visited:
        visited.add(tensor)
        if not tensor.ctx:
          topo.append(tensor)
          return
        for p in tensor.ctx.saved_tensors:
          build_topo(p)
        topo.append(tensor)
    build_topo(self) 
    for t in reversed(topo):
      assert t.grad, f'{t} has no grad'
      if not t.ctx:
        continue
      grads = t.cls.backward(t.ctx, t.grad)
      grads = grads if len(t.ctx.saved_tensors) > 1 else [grads]
      for t0, g in zip(t.ctx.saved_tensors, grads):
        t0.grad = g if t0.grad == None else t0.grad + g

  def _strides(self) -> None:
    self.strides = list(accumulate(self.shape[-1:0:-1], func=operator.mul, initial=(1 if len(self.shape)else None)))[::-1]
    return self.strides
  
  def item(self) -> Num:
    if len(self.shape): raise RuntimeError(f'a Tensor with {self.numel} elements cannot be converted to Scalar')
    return self.data

  def __getitem__(self, key) -> Self:
    # TODO: Remove dimensions when indexing down from NDim to MDim (m < n)
    # i.e.) indexing x[0,0,0] x.shape=(2,2,2) should return a scalar view of x
    if not len(self.shape): raise IndexError('invalid index of a 0-dim tensor. Use `tensor.item()`')
    x = Tensor(self.shape, self.data)
    x.index_view = to_nested_list(self, key)
    return x
    
  def broadcast_to(self: Self, broadcast_shape: Shape) -> Self:
    if self.shape == broadcast_shape:
      return self
    pad_s = pad_left(self.shape, broadcast_shape)
    # Set shape to original size with 1s padded for broadcasting
    x = self.reshape(*pad_s[0]) 
    return x.expand(*broadcast_shape)

  def __broadcast(self: Self, other: Self):
    assert self.ndim != 0 and other.ndim != 0, 'invalid broadcasting with scalar'
    new_shapes = pad_left(self.shape, other.shape)
    assert all(x == y or x == 1 or y == 1 for x, y in zip(*new_shapes)), 'invalid shapes for broadcasting {self.shape} and {other.shape}'
    bs = broadcast_shape(*new_shapes)
    a = self.broadcast_to(bs) 
    b = other.broadcast_to(bs)
    return a,b
  
  def __mul__(self, other: Self) -> Self:
    from shrimpgrad.autograd.function import Mul
    if self.is_scalar():
      other = other if isinstance(
        other, Tensor) else Tensor((), other)
      return Mul.apply(self, other)
    a, b = self.__broadcast(other)
    return Mul.apply(a,b)
  
  def __rmul__(self, other):
    return self * other

  def __add__(self, other: Self) -> Self:
    from shrimpgrad.autograd.function import Add 
    if self.is_scalar():
      other = other if isinstance(
        other, Tensor) else Tensor((), other)
      return Add.apply(self, other)
    a, b = self.__broadcast(other)
    return Add.apply(a,b)
  
  def __radd__(self, other):
    return self + other
  
  def __neg__(self):
    from shrimpgrad.autograd.function import Mul
    if self.is_scalar():
      return self * -1
    a, b = self.__broadcast(Tensor((1,), [-1.0]))
    return Mul.apply(a,b)
  
  def __sub__(self, other):
    if self.is_scalar():
      other = other if isinstance(
        other, Tensor) else Tensor((), other)
      return self + (-other) 
    return self + (-other) 

  def __rsub__(self, other):
    return self + (-other)
  
  def __truediv__(self, other):
    from shrimpgrad.autograd.function import Div
    other = other if isinstance(
      other, Tensor) else Tensor((), other)
    if self.is_scalar():
      return Div.apply(self, other)
    a, b = self.__broadcast(other)
    return Div.apply(a, b) 
  
  def __rtruediv__(self, other):
    if self.is_scalar():
      other = Tensor((), other)
      return other / self 
    other = Tensor((1,), [other]) 
    return other / self 
  
  def log(self) -> Self:
    from shrimpgrad.autograd.function import Log
    return Log.apply(self) 

  def relu(self) -> Self:
    from shrimpgrad.autograd.function import ReLU 
    return ReLU.apply(self)
    
  def __matmul__(self, other) -> Self:
    return self.matmul(other)

  def matmul(self, other: Self, reverse=False) -> Self:
    return other.dot(self) if reverse else self.dot(other) 
  
  def _canonicalize_axis(self, axis):
    return tuple(ax if ax >= 0 else ax + self.ndim for ax in (axis if isinstance(axis, Tuple) else (axis,))) 

  def mean(self, axis=None) -> Self:
    axis = axis if axis else tuple(i for i in range(self.ndim))
    axis_ = self._canonicalize_axis(axis)
    return  self.sum(axis=axis) / prod([self.shape[i] for i in axis_])

  def sum(self, axis:Union[int|Tuple[int,...]]=0, keepdim=False) -> Self:
    from shrimpgrad.autograd.function import Sum 
    axis_ = self._canonicalize_axis(axis) 
    shape = tuple(s for i, s in enumerate(self.shape) if i not in axis_)
    ret = Sum.apply(self, axis=axis_, keepdim=keepdim) 
    return ret if keepdim else ret.reshape(*shape)
  
  def dot(self, w) -> Self:
    # From https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py 
    n1, n2 = len(self.shape), len(w.shape)
    assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
    assert (L:=self.shape[-1]) == (R:=w.shape[-min(n2, 2)]), f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({L} != {R})"
    x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
    return (x*w).sum(axis=-1)
  
  def const(self, val:Num, **kwargs) -> Self:
    return Tensor.full_like(self, val, **kwargs)

  def expand(self, *shps) -> Self:
    from shrimpgrad.autograd.function import Expand 
    return Expand.apply(self, shape=tuple(shps))

  def reshape(self, *shps) -> Self:
    from shrimpgrad.autograd.function import Reshape  
    return Reshape.apply(self, shape=tuple(shps))

  def permute(self, order: Tuple[int,...]) -> Self:
    from shrimpgrad.autograd.function import Permute
    return Permute.apply(self, order=order)

  def transpose(self, ax0=1, ax1=0):
    ax0, ax1 = (ax0 + self.ndim if ax0 < 0 else ax0), (ax1 + self.ndim if ax1 < 0 else ax1)
    order = [i for i in range(self.ndim)]
    order[ax0], order[ax1] = order[ax1], order[ax0]
    return self.permute(order) 
  
  def linear(self, w: Tensor, bias:Optional[Tensor]=None) -> Tensor:
    return self.dot(w) + bias if bias else self.dot(w) 
  
  def exp(self):
    from shrimpgrad.autograd.function import Exp
    return Exp.apply(self)

  # Loss Functions
  def mse(self, y: Tensor):
    return (self-y)**2

  def __repr__(self): 
    if self.is_scalar(): return f'tensor({self.data})'
    return f'tensor({pformat(to_nested_list(self, None) if not self.index_view else self.index_view, width=40)})'
  def __str__(self): return self.__repr__()

  @staticmethod
  def zeros(shape: Shape, dtype:DType=dtypes.float32, **kwargs) -> Self: 
    return Tensor.full(shape, fill_value=0.0, dtype=dtype, **kwargs)

  @staticmethod
  def ones(shape: Shape, dtype:DType=dtypes.float32, **kwargs) -> Self: 
    return Tensor.full(shape, fill_value=1.0, dtype=dtype, **kwargs)

  @staticmethod
  def arange(start: int, stop:int, step:int=1, dtype:DType=dtypes.float32, **kwargs) -> Self: return Tensor(((stop - start) // step,), [float(i) if dtype == dtypes.float32 else int(i) for i in range(start, stop, step)], dtype, **kwargs) 

  @staticmethod
  def fromlist(shape: Shape, data:List[Num], dtype=dtypes.float32, **kwargs):
    return Tensor(shape, data=data, dtype=dtype, **kwargs)

  @staticmethod
  def full(shape: Shape, fill_value: Num, dtype=dtypes.float32, **kwargs) -> Tensor:
    if not len(shape): return Tensor((), fill_value)
    return Tensor(shape, [float(fill_value) if dtype == dtypes.float32 else int(fill_value)]*prod(shape), **kwargs)

  @staticmethod
  def full_like(t: Tensor, fill_value: Num, **kwargs) -> Tensor:
    return Tensor.full(t.shape, fill_value=fill_value, dtype=t.dtype, **kwargs)
  
  @staticmethod
  def zeros_like(t: Tensor, **kwargs) -> Tensor:
    return Tensor.full_like(t, 0.0, **kwargs)
  
  @staticmethod
  def ones_like(t: Tensor, **kwargs) -> Tensor:
    return Tensor.full_like(t, 1.0, **kwargs)
  
  @staticmethod
  def eye(n: int, dtype=dtypes.float32, **kwargs) -> Tensor:
    assert n > 0, 'identity matrix requires dimension > 0' 
    data = [0.0] * (n**2) 
    for i in range(n):
      data[i*n + i] = 1.0
    return Tensor((n,n), data, dtype, **kwargs)
  
  @staticmethod
  def rand(*shape, dtype=dtypes.float32, **kwargs) -> Tensor:
    # TODO: Change to non lib based threefry or philox
    return Tensor.uniform(*shape, low=0, high=1, dtype=dtype, **kwargs) 

  @staticmethod
  def randn(*shape, dtype=dtypes.float32, **kwargs) -> Tensor:
    #TODO: Box Muller Transform 
    return Tensor(shape, [gauss(0, 1) for _ in range(prod(shape))], dtype, **kwargs)

  @staticmethod
  def uniform(*shape, low:Union[int, float]=0, high:Union[int, float]=10, dtype=dtypes.float32, **kwargs) -> Tensor:
    return Tensor(shape, [uniform(low, high) for _ in range(prod(shape))], dtype=dtype, **kwargs)

  @staticmethod
  def kaiming_uniform(*shape, mode:str='fan_in', nonlinearity:str='leaky_relu', a=0.1, **kwargs) -> Tensor:
    bound = math.sqrt(3.0) * calc_gain(a) / calc_fan_in_fan_out(shape)[0]
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)
  
  @staticmethod
  def scalar(x: int|float) -> Self:
    return Tensor((), data=x)
  
  # Niceties
  def size(self, dim:int|None=None) -> Tuple[int,...]|int:
    assert dim == None or 0 <= dim < self.ndim, f'invalid dimension {dim} for tensor with shape of {self.ndim}-d'
    if dim: return self.shape[dim]
    return tuple(self.shape)
 