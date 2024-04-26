from __future__ import annotations
from typing import Callable, Iterable, Optional, Self, Type, TypeAlias, Union, List, Tuple
from functools import reduce
import operator
from pprint import pformat
from shrimpgrad.cblas import sgemm
from shrimpgrad.dtype import DType, dtypes
from random import uniform, gauss

Num: TypeAlias = Union[float, int, complex]
Shape: TypeAlias = Tuple[int, ...]

def prod(x: Iterable[int|float]) -> Union[float, int]: return reduce(operator.mul, x, 1)
def pad_left(*shps: Tuple[int, ...], v=1) -> List[Tuple[int ,...]]: return [tuple((v,)*(max(len(s) for s in shps)-len(s)) + s) for s in shps]
def broadcast_shape(*shps: Tuple[int, ...]) -> Tuple[int, ...]: return tuple([max([s[dim] for s in shps]) for dim in range(len(shps[0]))])
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

class Function:
  def __init__(self, device, *tensors):
    # TODO: Make sure require_grad are respected and set properly
    self.device = device
    self.parents = tensors
    for p in self.parents:
      # Init parent tensors to zero if not init
      if not p.grad:
        p.grad = Tensor.zeros(p.shape, p.dtype) 
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    self.out = None
  
  def forward(self, *args, **kwargs): raise NotImplementedError(f'forward is not implemented for {type(self)}')
  def backward(self, *args, **kwargs): raise RuntimeError(f'backward not implemented for {type(self)}')

  @classmethod
  def apply(fn: Type[Function], *tensors, **kwargs) -> Tensor:
    ctx = fn(tensors[0].device, *tensors)
    ret = ctx.forward(*tensors, **kwargs)
    ret.ctx = ctx
    ret.requires_grad = ctx.requires_grad
    return ret

def binary_op(F: Callable, a: Tensor, b: Tensor, dim:int, off_a:int, off_b:int, loops:Iterable[Tuple], result:Iterable[Union[int, float, complex]]) -> None:
  if not loops:  return 
  s, e, step = loops[0]
  for i in range(s, e, step):
    if len(loops) == 1: result.append(F(a.data[off_a + i*step*a.strides[dim]] , b.data[off_b + i*step*b.strides[dim]]))
    else: binary_op(F, a, b, dim+1, off_a + i*a.strides[dim]*step, off_b + i*b.strides[dim]*step, loops[1:], result)
  return 

def unary_op(F: Callable, a: Tensor, dim: int, off: int, loops: Iterable[Tuple], result: Iterable[Union[Num]]) -> None:
  if not loops: return
  s, e, step = loops[0]
  for i in range(s, e, step):
    if len(loops) == 1: result.append(F(a.data[off + i*step*a.strides[dim]]))
    else: unary_op(F, a, dim+1, off + i*a.strides[dim]*step, loops[1:], result)
  return 

def reduce_(F: Callable, x: Tensor, loops, off=0, dim=0, ax=0, keepdims=False):
  assert x.ndim > 0, 'scalars cannot be reduced'
  ax = ax + x.ndim if ax < 0 else ax
  assert ax < x.ndim, f'axis={ax} is out of bounds for tensor with shape={x.shape}'
  s,e,step = loops[0]
  res, accum = [], []
  for _ in range(s,e,step):
    if ax == dim: accum.append(x.data[off:off+x.strides[dim]])
    else: res += reduce_(F, x, loops[1:], off, dim+1, ax, keepdims)
    off += x.strides[dim]
  return [reduce(F,r) for r in zip(*accum)] if accum else res 

class Add(Function):
  def forward(self, a: Tensor, b: Tensor) -> Tensor:
    if a.is_scalar() and b.is_scalar():
      self.out = Tensor((), a.data + b.data)
      return self.out
    result = []
    binary_op(operator.add, a,b, 0, 0,0, a.calc_loops(None), result) 
    self.out = Tensor(a.shape, result, dtype=a.dtype)
    return self.out

  def backward(self):
    a = self.parents[0]
    b = self.parents[1]
    a.grad += self.out.grad 
    b.grad += self.out.grad

class Mul(Function):
  def forward(self, a: Tensor, b: Tensor) -> Tensor:
    if a.is_scalar() and b.is_scalar():
      self.out = Tensor((), a.data * b.data)
      return self.out
    result = []
    binary_op(operator.mul, a,b, 0, 0,0, a.calc_loops(None), result) 
    self.out = Tensor(a.shape, result, dtype=a.dtype)
    return self.out

  def backward(self):
    a = self.parents[0]
    b = self.parents[1]
    grad_out = self.out.grad
    grad_a = b * grad_out
    grad_b = a * grad_out

    a.grad += grad_a 
    b.grad += grad_b

class ReLU(Function):
  def forward(self, a: Tensor) -> Tensor:
    if a.is_scalar():
      self.out = Tensor((), (0 if a.data < 0 else a.data))
      return self.out

    result = []
    unary_op(lambda x: 0.0 if x < 0 else x, a, 0, 0, a.calc_loops(None), result)
    self.out = Tensor(a.shape, result, dtype=a.dtype)
    return self.out
  
  def backward(self):
    a = self.parents[0]
    if a.is_scalar():
      a.grad += (a.data > 0) * self.out.grad
      return

    result = []
    unary_op(lambda x: 1 if x > 0 else 0, a,  0, 0, a.calc_loops(None), result)
    x = Tensor(a.shape, result, dtype=a.dtype, requires_grad=False) 
    a.grad += x * a * self.out.grad

class Pow(Function):
  def forward(self, a:Tensor, b: Tensor) -> Tensor:
    if a.is_scalar():
      self.out =  Tensor((), a.data ** b.data)
      return self.out
    
    result = []
    binary_op(operator.pow, a, b, 0, 0, 0, a.calc_loops(None), result)
    self.out = Tensor(a.shape, result, dtype=a.dtype)

    return self.out
  
  def backward(self):
    a, b = self.parents[0], self.parents[1]
    if a.is_scalar():
      a.grad += b * (a ** (b - 1)) * self.out.grad
    else:
      a.grad += b * (a ** (b - Tensor((1,), [1]))) * self.out.grad

class Sum(Function):
  def forward(self, x: Tensor, axis: int=0, keepdim=False) -> Tensor:
    ret = reduce_(operator.add, x, x.calc_loops(None), 0, 0, ax=axis)
    self.out = Tensor((*x.shape[0:axis],*[1]*(keepdim), *x.shape[axis+1:]), ret)
    return self.out

  def backward(self):
    x = self.parents[0]
    x.grad = self.out.grad.broadcast_to(x.shape)

class Reshape(Function):
  def forward(self, x: Tensor, shape: Tuple[int,...] ) -> Tensor:
    if prod(shape) != x.size: raise RuntimeError('shape \'{shape}\' is invalid for input of size {x.size}')
    self.out = Tensor(shape, x.data, dtype=x.dtype)
    return self.out

  def backward(self):
    x = self.parents[0]
    return self.out.grad.reshape(*x.shape)

class Permute(Function):
  def forward(self, x: Tensor, order: Tuple[int, ...]) -> Tensor:
    self.order = order
    new_shape = [x.shape[i] for i in order]
    new_strides = [x.strides[i] for i in order]
    self.out = Tensor(tuple(new_shape), x.data, dtype=x.dtype) 
    self.out.strides = new_strides
    return self.out
  
  def backward(self):
    return self.out.grad.permute(argsort(self.order))

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
    def build_topo(tensor: Tensor) -> List[Tensor]:  
      if tensor not in visited:
        visited.add(tensor)
        if not tensor.ctx:
          topo.append(tensor)
          return
        for p in tensor.ctx.parents:
          build_topo(p)
        topo.append(tensor)
    build_topo(self) 
    for t in reversed(topo):
      if not t.ctx:
        continue
      t.ctx.backward()

  def __calc_strides(self) -> None:
    self.strides.clear()
    out: int = 1
    for dim in self.shape:
      out *= dim
      self.strides.append(self.size // out)
  
  def calc_loops(self, key: Optional[slice|int]) -> Iterable[int]:
    if not key and not isinstance(key, int):  key = [slice(0, dim, 1) for dim in self.shape]
    if isinstance(key, int) or isinstance(key, slice): key = (key,)
    if len(key) > len(self.shape): raise IndexError(f'index of {key} is larger than dim {self.shape}.')
    extra_dim = 0
    if len(key) < len(self.shape): extra_dim = len(self.shape) - len(key)
    loops = []
    for i, k in enumerate(key):
      if isinstance(k, int):
        if k >= self.shape[i]:  raise IndexError(f'index of {k} is out of bounds for dim with size {self.shape[i]}')
        start = k
        if start < 0:
          if abs(start) > self.shape[i]:
            raise IndexError(f'index of {start} is out of bounds for dim with size {self.shape[i]}')
          k = self.shape[i] + start
        start, end = k, k + 1
        loops.append((start,end, 1))
      elif isinstance(k, slice):
        start, end, step = k.indices(self.shape[i])
        if start < 0:
          if abs(start) > self.shape[i]:
            raise IndexError(f'index of {start} is out of bounds for dim with size {self.shape[i]}')
          start = self.shape[i] + start 
        if end < 0:
          if abs(end) > self.shape[i]:
            raise IndexError(f'index of {end} is out of bounds for dim with size {self.shape[i]}')
          end = self.shape[i] + end
        if start >= end: return [] 
        loops.append((start, end, step))
    if extra_dim: 
      for ed in range(len(key), len(key) + extra_dim): loops.append((0, self.shape[ed], 1))
    return loops

  def __build_view(self, key: Optional[slice|int]) -> Iterable:
    def build(dim:int, offset:int, loops:Iterable[tuple], tensor:Iterable[Num]) -> Iterable[Num]:
      if not loops: return tensor
      s, e, step = loops[0]
      for i in range(s, e, step):
        if len(loops) == 1: 
          tensor.append(self.data[offset+i*step*self.strides[dim]])
        else: 
          tensor.append([])
          build(dim+1, offset + i*self.strides[dim]*step, loops[1:], tensor[-1])
      return tensor 
    return build(0, 0, self.calc_loops(key), []) 
  
  def item(self) -> Num:
    if len(self.shape): raise RuntimeError(f'a Tensor with {self.size} elements cannot be converted to Scalar')
    return self.data

  def __getitem__(self, key) -> Self:
    if not len(self.shape): raise IndexError('invalid index of a 0-dim tensor. Use `tensor.item()`')
    x = Tensor(self.shape, self.data)
    x.index_view = x.__build_view(key)
    return x
    
  def broadcast_to(self: Self, broadcast_shape: Shape) -> Self:
    if self.shape == broadcast_shape:
      return self
    pad_s = pad_left(self.shape, broadcast_shape)
    # Set shape to original size with 1s padded for broadcasting
    nt = Tensor(pad_s[0], self.data, self.dtype)
    # Where the shape is 1, change the stride to 0
    for i, v in enumerate(pad_s[0]): 
      if v == 1: nt.strides[i] = 0
    # Set the shape to the broadcast shape
    nt.shape = broadcast_shape
    return nt

  def __broadcast(self: Self, other: Self):
    assert self.ndim != 0 and other.ndim != 0, 'invalid broadcasting with scalar'
    new_shapes = pad_left(self.shape, other.shape)
    assert all(x == y or x == 1 or y == 1 for x, y in zip(*new_shapes)), 'invalid shapes for broadcasting {self.shape} and {other.shape}'
    bs = broadcast_shape(*new_shapes)
    a = self.broadcast_to(bs) 
    b = other.broadcast_to(bs)
    return a,b
  
  def __mul__(self, other: Self) -> Self:
    if self.is_scalar():
      other = other if isinstance(
        other, Tensor) else Tensor((), other)
      return Mul.apply(self, other)
    a, b = self.__broadcast(other)
    return Mul.apply(a,b)
  
  def __rmul__(self, other):
    return self * other

  def __add__(self, other: Self) -> Self:
    if self.is_scalar():
      other = other if isinstance(
        other, Tensor) else Tensor((), other)
      return Add.apply(self, other)
    a, b = self.__broadcast(other)
    return Add.apply(a,b)
  
  def __radd__(self, other):
    return self + other
  
  def __neg__(self):
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
  
  def __pow__(self, other):
    if self.is_scalar():
      if isinstance(other, Tensor):
        assert other.ndim == 0, 'scalar to power of {other.dim} is undefined'
      other = other if isinstance(
        other, Tensor) else Tensor((), other)
      return Pow.apply(self, other) 
    if isinstance(other, Num):
      return self ** Tensor((1,), [other])
    if other.is_scalar():
      return self ** other.data
    a,b = self.__broadcast(other)
    return Pow.apply(a, b)
    
  def __truediv__(self, other):
    return self * (other ** -1)
  
  def __rtruediv__(self, other):
    if self.is_scalar():
      other = Tensor((), other)
      return other * (self ** -1)
    other = Tensor((1,), [other]) 
    return other * (self ** -1)

  def relu(self) -> Self:
    return ReLU.apply(self)
    
  def __matmul__(self, other) -> Self:
    return self.matmul(other)

  def matmul(self, other: Self) -> Self:
    # 1D x 1D (dot product) 
    if len(self.shape) == 1 and len(other.shape) == 1:
      if self.shape[0] != other.shape[0]: raise RuntimeError(f'inconsistent tensor size, expected tensor [{self.shape[0]}] and src [{other.shape[0]}] to have the same number of elements, but got {self.shape[0]} and {other.shape[0]} elements respectively')
      return Tensor((), sum(map(lambda x: x[0]*x[1], zip(self.data, other.data))))

    # Classic NxM * MxN matrix mult
    if len(self.shape) == 2 and len(other.shape) == 2:
      if self.shape[1] != other.shape[0]: raise RuntimeError('mat1 and mat2 shapes cannot be multiplied ({self.shape[0]}x{self.shape[1]} and {other.shape[0]}x{other.shape[1]})')
      result = sgemm(self, other)
      return Tensor((self.shape[0], other.shape[1]), [x for x in result])
    
  def sum(self, axis=0, keepdim=False) -> Self:
    return Sum.apply(self, axis=axis if axis >= 0 else axis + self.ndim, keepdim=keepdim) 
  
  def dot(self, w) -> Self:
    assert self.ndim > 0 or w.ndim > 0, f'dot works only on tensors of 1D and higher not scalars: {self.shape}, {w.shape}'
    assert self.shape[-1] == w.shape[-2] if w.ndim > 1 else True, f'last index of {self.shape} != 2nd to last index of {w.shape}' 
    x = self.reshape(*self.shape[0:-1], *[1]*min(self.ndim-1, w.ndim-1, 1), self.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(self.ndim-1, w.ndim-1, 1), *w.shape[-min(w.ndim, 2):]).transpose(-1, -min(w.ndim, 2)) 
    z = (x*w)
    return z.sum(axis=-1)

  def reshape(self, *shps) -> Self: 
    return Reshape.apply(self, shape=tuple(shps))

  def permute(self, order: Tuple[int,...]) -> Self:
    return Permute.apply(self, order=order)

  def transpose(self, ax0=1, ax1=0):
    ax0, ax1 = (ax0 + self.ndim if ax0 < 0 else ax0), (ax1 + self.ndim if ax1 < 0 else ax1)
    order = [i for i in range(self.ndim)]
    order[ax0], order[ax1] = order[ax1], order[ax0]
    return self.permute(order) 

  def __repr__(self): 
    if self.is_scalar(): return f'tensor({self.data})'
    return f'tensor({pformat(self.__build_view(None) if not self.index_view else self.index_view, width=40)})'
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



