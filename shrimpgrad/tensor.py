from __future__ import annotations
import ctypes
import functools
import math
from typing import Callable, List, Optional, TypeAlias, Union, Tuple
from shrimpgrad.dtype import DType, dtypes, ConstType
from random import uniform, gauss
from shrimpgrad.engine.runner import realize
from shrimpgrad.future import Thunk
from shrimpgrad.runtime.clang import ClangDevice
from shrimpgrad.util import calc_fan_in_fan_out, calc_gain, prod
import numpy as np

Num: TypeAlias = Union[float, int, complex]
Shape: TypeAlias = Tuple[int, ...]

def pad_left(*shps: Tuple[int, ...], v=1) -> List[Tuple[int ,...]]: return [tuple((v,)*(max(len(s) for s in shps)-len(s)) + s) for s in shps]
def broadcast_shape(*shps: Tuple[int, ...]) -> Tuple[int, ...]: return tuple([max([s[dim] for s in shps]) for dim in range(len(shps[0]))])

class Tensor:
  def __init__(self, shape: Shape, data: Union[List, bytes, ConstType, Thunk], dtype:DType=dtypes.float32, device=ClangDevice(), requires_grad:Optional[bool]=None) -> Tensor:
    self.requires_grad, self.index_view = requires_grad, None
    self.grad: Optional[Tensor] = None
    from shrimpgrad.autograd.function import Function
    self.ctx: Optional[Function] = None
    if isinstance(data, Thunk): self.thunk = data
    else:
      if shape == () and not isinstance(data, ConstType) and len(data) == 1: data = data[0]
      self.thunk = Thunk.load_from_cpu(data, dtype, shape)
    if self.thunk.device != device: self.thunk = self.thunk.copy_to_device(device)

  def backward(self) -> Tensor:
    self.grad = Tensor.ones(self.shape, self.dtype)
    visited = set()
    topo = []
    # TODO: Turn this into generator so we don't allocate memory for
    # a massive list
    def build_topo(tensor: Tensor) -> List[Tensor]:
      if tensor not in visited:
        visited.add(tensor)
        if not tensor.ctx: return
        for p in tensor.ctx.tensors: build_topo(p)
        topo.append(tensor)
    build_topo(self)
    for t in reversed(topo):
      assert t.grad, f'{t} has no gradient'
      grads = t.cls.backward(t.ctx, t.grad.thunk)
      grads = [Tensor(g.shape, g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t.ctx.tensors) == 1 else grads)]
      for t0, g in zip(t.ctx.tensors, grads):
        t0.grad = g if t0.grad is None else t0.grad + g
    return self

  @property
  def shape(self): return self.thunk.shape
  @property
  def dtype(self): return self.thunk.dtype
  @property
  def numel(self): return self.thunk.numel
  @property
  def device(self): return self.thunk.device
  @property
  def ndim(self): return self.thunk.ndim

  # Indexing
  def item(self) -> ConstType:
    if len(self.shape): raise RuntimeError(f'a Tensor with {self.numel} elements cannot be converted to Scalar')
    return self.thunk

  def __getitem__(self, key) -> Tensor:
    # TODO: Remove dimensions when indexing down from NDim to MDim (m < n)
    # i.e.) indexing x[0,0,0] x.shape=(2,2,2) should return a scalar view of x
    if not len(self.shape): raise IndexError('invalid index of a 0-dim tensor. Use `tensor.item()`')
    x = Tensor(self.shape, self.thunk)
    return x

  # Broadcasting, Assignment, Casting and Data Augmentation
  def broadcast_to(self: Tensor, broadcast_shape: Shape) -> Tensor:
    if self.shape == broadcast_shape:
      return self
    pad_s = pad_left(self.shape, broadcast_shape)
    # Set shape to original size with 1s padded for broadcasting (unless padding had no effect)
    x = self.reshape(*pad_s[0]) if pad_s[0] != self.shape else self
    return x.expand(*broadcast_shape)

  def __broadcast(self, y: Union[Tensor, ConstType], reverse=False) -> Tuple[Tensor]:
    x = self
    if not isinstance(y, Tensor):
      assert isinstance(y, ConstType), f'type(y)={type(y)} is not a ConstType'
      y = Tensor((), data=y, dtype=dtypes.from_py(y))
    new_shapes = pad_left(self.shape, y.shape)
    assert all(x == y or x == 1 or y == 1 for x, y in zip(*new_shapes)), f'invalid shapes for broadcasting {self.shape} and {y.shape}'
    bs = broadcast_shape(*new_shapes)
    if reverse: x, y = y, x
    return x.broadcast_to(bs), y.broadcast_to(bs)

  def replace(self, x: Tensor) -> Tensor:
    assert x.shape == self.shape, f'shape mismatch on replace {self.shape} != {x.shape}'
    assert x.dtype == self.dtype, f'dtype mismatch on replace {self.dtype} != {x.dtype}'
    self.thunk = x.thunk
    return self

  def assign(self, x: Tensor) -> Tensor:
    assert x.shape == self.shape, f'shape mismatch on assign {self.shape} != {x.shape}'
    assert x.dtype == self.dtype, f'dtype mismatch on assign {self.dtype} != {x.dtype}'
    return self.replace(x)

  def cast(self, dtype: DType) -> Tensor:
    from shrimpgrad.autograd.function import Cast
    return Cast.apply(self, dtype=dtype)

  def clamp(self, min_val=0.0, max_val=1.0) -> Tensor:
    return self.where((self.detach() > min_val), min_val).where(self.detach() < max_val, max_val)

  def where(self, condition: Tensor, y: Tensor ) -> Tensor:
    from shrimpgrad.autograd.function import Where
    cond, x = condition.__broadcast(self)
    cond, y_ = cond.__broadcast(y)
    return Where.apply(cond.cast(dtypes.bool), *x.__broadcast(y_))

  def detach(self) -> Tensor:
    return Tensor(self.shape, self.thunk, self.dtype, requires_grad=False)

  # Arithmetic and Logical Functions
  def mul(self, other, reverse=False) -> Tensor:
    from shrimpgrad.autograd.function import Mul
    return Mul.apply(*self.__broadcast(other, reverse))

  def add(self, other, reverse=False) -> Tensor:
    from shrimpgrad.autograd.function import Add
    return Add.apply(*self.__broadcast(other, reverse))

  def sub(self, other, reverse=False) -> Tensor:
    from shrimpgrad.autograd.function import Sub
    return Sub.apply(*self.__broadcast(other, reverse))

  def div(self, other, reverse=False) -> Tensor:
    from shrimpgrad.autograd.function import Div
    return Div.apply(*self.__broadcast(other, reverse))

  def exp(self):
    from shrimpgrad.autograd.function import Exp
    return Exp.apply(self)

  def log(self) -> Tensor:
    from shrimpgrad.autograd.function import Log
    return Log.apply(self)

  def square(self) -> Tensor: return self * self

  def _canonicalize_axis(self, axis):
    return tuple(ax if ax >= 0 else ax + self.ndim for ax in (axis if isinstance(axis, Tuple) else (axis,)))

  def mean(self, axis=None) -> Tensor:
    axis = axis if axis else tuple(i for i in range(self.ndim))
    axis_ = self._canonicalize_axis(axis)
    return  self.sum(axis=axis) / prod([self.shape[i] for i in axis_])

  def sum(self, axis:Union[int|Tuple[int,...]]=None, keepdim=False) -> Tensor:
    from shrimpgrad.autograd.function import Sum
    axis = axis if axis != None else tuple(i for i in range(self.ndim))
    axis_ = self._canonicalize_axis(axis)
    shape = tuple(s for i, s in enumerate(self.shape) if i not in axis_)
    ret = Sum.apply(self, axis=axis_)
    return ret if keepdim else ret.reshape(*shape)

  def dot(self, w) -> Tensor:
    # From https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
    n1, n2 = len(self.shape), len(w.shape)
    assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
    assert (L:=self.shape[-1]) == (R:=w.shape[-min(n2, 2)]), f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({L} != {R})"
    x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
    return (x*w).sum(axis=-1)

  def matmul(self, other: Tensor, reverse=False) -> Tensor:
    return other.dot(self) if reverse else self.dot(other)

  def logical_not(self) -> Tensor:
    from shrimpgrad.autograd.function import Eq
    return Eq.apply(*self.__broadcast(False))

  def __lt__(self, x) -> Tensor:
    from shrimpgrad.autograd.function import Less
    return Less.apply(*self.__broadcast(x, False))

  def __gt__(self, x) -> Tensor:
    from shrimpgrad.autograd.function import Less
    return Less.apply(*self.__broadcast(x, True))

  def __eq__(self, x) -> Tensor:
    from shrimpgrad.autograd.function import Eq
    return Eq.apply(*self.__broadcast(x))

  def __ge__(self, x) -> Tensor: return (self<x).logical_not()
  def __le__(self, x) -> Tensor: return (self>x).logical_not()
  def __ne__(self, x) -> Tensor: return (self==x).logical_not()
  def __mul__(self, other: Tensor) -> Tensor: return self.mul(other)
  def __rmul__(self, other): return self.mul(other, reverse=True)
  def __add__(self, other: Tensor) -> Tensor: return self.add(other)
  def __radd__(self, other): return self.add(other, reverse=True)
  def __neg__(self): return self.mul(-1)
  def __sub__(self, other): return self.sub(other)
  def __rsub__(self, other): return self.sub(other, True)
  def __truediv__(self, other): return self.div(other)
  def __rtruediv__(self, other): return self.div(other, reverse=True)
  def __matmul__(self, other) -> Tensor: return self.matmul(other)

  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))

  # Activation Functions
  def relu(self) -> Tensor:
    from shrimpgrad.autograd.function import ReLU
    return ReLU.apply(self)
  def sigmoid(self) -> Tensor:
    from shrimpgrad.autograd.function import Sigmoid
    return Sigmoid.apply(self)

  # Loss Functions
  def binary_cross_entropy(self, y: Tensor) -> Tensor: return ((-y)*self.log() - (1.0-y)*((1.0-self).log())).mean()

  def hinge_loss(self, target: Tensor) -> Tensor: return (1.0 + -target*self).relu().sum() * (1.0/target.shape[0])
  def mse(self, target: Tensor) -> Tensor: return (self-target).square().mean()

  # Shape Shift Functions
  def expand(self, *shps) -> Tensor:
    from shrimpgrad.autograd.function import Expand
    return Expand.apply(self, shape=tuple(shps))

  def reshape(self, *shps) -> Tensor:
    from shrimpgrad.autograd.function import Reshape
    return Reshape.apply(self, shape=tuple(shps))

  def permute(self, order: Tuple[int,...]) -> Tensor:
    from shrimpgrad.autograd.function import Permute
    return Permute.apply(self, order=order)

  def transpose(self, ax0=1, ax1=0):
    ax0, ax1 = (ax0 + self.ndim if ax0 < 0 else ax0), (ax1 + self.ndim if ax1 < 0 else ax1)
    order = [i for i in range(self.ndim)]
    order[ax0], order[ax1] = order[ax1], order[ax0]
    return self.permute(order)

  # Neural Network Layer Functions
  def linear(self, w: Tensor, bias:Optional[Tensor]=None) -> Tensor:
    return self.dot(w) + bias if bias else self.dot(w)

  # TODO: from tinygrad nice impl, see what we can change here
  def sequential(self, ll:List[Callable[[Tensor], Tensor]]): return functools.reduce(lambda x,f: f(x), ll, self)

  # Creation Functions
  @staticmethod
  def zeros(shape: Shape, dtype:DType=dtypes.float32, **kwargs) -> Tensor: return Tensor.full(shape, fill_value=0.0, dtype=dtype, **kwargs)

  @staticmethod
  def ones(shape: Shape, dtype:DType=dtypes.float32, **kwargs) -> Tensor:  return Tensor.full(shape, fill_value=1.0, dtype=dtype, **kwargs)

  @staticmethod
  def arange(start: int, stop:int, step:int=1, dtype:DType=dtypes.float32, **kwargs) -> Tensor: return Tensor(((stop - start) // step,), [float(i) if dtype == dtypes.float32 else int(i) for i in range(start, stop, step)], dtype, **kwargs)

  @staticmethod
  def fromlist(shape: Shape, data:List[Num], dtype=dtypes.float32, **kwargs): return Tensor(shape, data=data, dtype=dtype, **kwargs)

  @staticmethod
  def full(shape: Shape, fill_value: Num, dtype=dtypes.float32, **kwargs) -> Tensor:
    if not len(shape): return Tensor((), fill_value)
    return Tensor(shape, [float(fill_value) if dtype == dtypes.float32 else int(fill_value)]*prod(shape), **kwargs)

  @staticmethod
  def full_like(t: Tensor, fill_value: Num, **kwargs) -> Tensor: return Tensor.full(t.shape, fill_value=fill_value, dtype=t.dtype, **kwargs)

  @staticmethod
  def zeros_like(t: Tensor, **kwargs) -> Tensor: return Tensor.full_like(t, 0.0, **kwargs)

  @staticmethod
  def ones_like(t: Tensor, **kwargs) -> Tensor: return Tensor.full_like(t, 1.0, **kwargs)

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
    # TODO: Add support for other nonlinearties and fan_out
    bound = math.sqrt(3.0) * calc_gain(a) / calc_fan_in_fan_out(shape)[0]
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

  def const(self, val:Num, **kwargs) -> Tensor: return Tensor.full_like(self, val, **kwargs)

  @staticmethod
  def scalar(x: ConstType) -> Tensor: return Tensor((), data=x)

  # Nice to have
  def size(self, dim:int|None=None) -> Tuple[int,...]|int:
    assert dim == None or 0 <= dim < self.ndim, f'invalid dimension {dim} for tensor with shape of {self.ndim}-d'
    if not dim is None: return self.shape[dim]
    return tuple(self.shape)

  def is_scalar(self): return not self.ndim

  # Trigger evaluation
  def realize(self) -> Tensor:
    realize(self.thunk)
    return self

  # Extract data
  def data(self):
    # TODO: Change this to something worthy
    base = self.thunk.base
    if hasattr(base, 'buff'):
      data = base.buff.pointer(ctypes.c_float)
      if not self.thunk.vt.contiguous:
        strides = tuple([s*4 for s in self.thunk.vt.strides])
        arr = np.frombuffer(data, dtype=np.float32)
        arr = np.lib.stride_tricks.as_strided(arr, shape=self.shape, strides=strides)
        return arr
      if base.shape == ():
        return np.frombuffer(data, dtype=np.float32).reshape(())
    else:
      raise TypeError("self is not realized wheres the bufff")
    return np.frombuffer(data, dtype=np.float32).reshape(self.shape)

  def numpy(self): return self.realize().data()

  # Object Representation
  def analyze(self):
    assert self.thunk.base.realized is not None, "Can't analyze unrealized tensor."
    is_view = self.thunk.isview
    op = self.thunk._op
    buffer = self.thunk.base.buff
    buffer_addr = f"0x{ctypes.addressof(buffer._pointer(ctypes.c_float)):X}"
    grad_data = []
    grad_buffer_addr = None
    grad_alloc = False
    if self.grad is not None:
      grad_buffer = self.grad.thunk.base.buff
      grad_buffer_addr = f"0x{ctypes.addressof(grad_buffer._pointer(ctypes.c_float)):X}" if hasattr(grad_buffer, '_buf') else "NONE"
      if hasattr(grad_buffer, '_buf'):
        grad_data = self.grad.data().flatten()[0:5]
      grad_alloc = grad_buffer.allocated
    print(f"{op = } {is_view = } alloc={buffer.allocated} {buffer_addr = } alloc={grad_alloc} {grad_buffer_addr = } {grad_data =  }")
  

  def __repr__(self): return f"<Tensor {self.thunk!r} on {self.device} with grad {(self.grad.thunk if self.grad is not None else None)!r}>"
  def __str__(self): return self.__repr__()
  def __hash__(self): return id(self)