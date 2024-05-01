from __future__ import annotations
from typing import Iterable, List, Optional, Self, TypeAlias, Union, Tuple
from pprint import pformat
from shrimpgrad.dtype import DType, dtypes
from random import uniform, gauss
from shrimpgrad.util import prod, to_nested_list 

Num: TypeAlias = Union[float, int, complex]
Shape: TypeAlias = Tuple[int, ...]

def pad_left(*shps: Tuple[int, ...], v=1) -> List[Tuple[int ,...]]: return [tuple((v,)*(max(len(s) for s in shps)-len(s)) + s) for s in shps]
def broadcast_shape(*shps: Tuple[int, ...]) -> Tuple[int, ...]: return tuple([max([s[dim] for s in shps]) for dim in range(len(shps[0]))])

class Tensor:
  def __init__(self, shape: Shape, data: Union[Iterable[Num], Num], dtype:DType=dtypes.float32, device='CPU', requires_grad:Optional[bool]=None) -> Self:
    self.shape, self.data, self.size, self.strides, self.dtype = shape, data, prod(shape), [], dtype
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
      return
    self.__calc_strides()
  
  def is_scalar(self):
    return not self.ndim 

  def backward(self) -> Tensor:
    self.grad = Tensor.ones(self.shape, self.dtype)
    visited = set()
    topo = []
    # TODO: Turn this into generator so we don't generate
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
      print(t.cls)
      grads = t.cls.backward(t.ctx, t.grad)
      grads = grads if len(t.ctx.saved_tensors) > 1 else [grads]
      for t0, g in zip(t.ctx.saved_tensors, grads):
        t0.grad = g if t0.grad == None else t0.grad + g

  def __calc_strides(self) -> None:
    self.strides.clear()
    out: int = 1
    for dim in self.shape:
      out *= dim
      self.strides.append(self.size // out)
  
  def item(self) -> Num:
    if len(self.shape): raise RuntimeError(f'a Tensor with {self.size} elements cannot be converted to Scalar')
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
    nt = Tensor(pad_s[0], self.data, self.dtype)
    # pad prev stride to new length and copy original dim strides to nt
    pad_strd = pad_left(tuple(self.strides), broadcast_shape)[0]
    for i in range(len(pad_strd)-1, len(pad_strd) - len(self.strides), -1):
      nt.strides[i] = pad_strd[i]
    # Where the shape is 1, change the stride to 0
    for i, v in enumerate(pad_s[0]): 
      if v == 1: nt.strides[i] = 0
    # Set the shape to the broadcast shape
    nt.shape = broadcast_shape
    # TODO: Need to use reshapes and expands here otherwise the graph disconnects
    # on broadcasting via the return of nt
    self.shape = broadcast_shape
    self.strides = nt.strides
    return self 

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

  def relu(self) -> Self:
    from shrimpgrad.autograd.function import ReLU 
    return ReLU.apply(self)
    
  def __matmul__(self, other) -> Self:
    return self.matmul(other)

  def matmul(self, other: Self, reverse=False) -> Self:
    return other.dot(self) if reverse else self.dot(other) 

  def sum(self, axis=0, keepdim=False) -> Self:
    from shrimpgrad.autograd.function import Sum 
    axis_ = axis if axis >= 0 else axis + self.ndim
    shape = tuple(s for i, s in enumerate(self.shape) if axis_ != i)
    ret = Sum.apply(self, axis=axis_, keepdim=keepdim) 
    return ret if keepdim else ret.reshape(*shape)
  
  def dot(self, w) -> Self:
    # From https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py 
    n1, n2 = len(self.shape), len(w.shape)
    assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
    assert (L:=self.shape[-1]) == (R:=w.shape[-min(n2, 2)]), f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({L} != {R})"
    x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
    z = x*w
    return z.sum(axis=-1)

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



