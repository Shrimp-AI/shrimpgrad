from __future__ import annotations
import ctypes
import functools
import math
from typing import Callable, List, Optional, Type, Union, Tuple
from shrimpgrad.device import Device
from shrimpgrad.dtype import DType, dtypes, ConstType, to_ctype, to_numpy
from random import uniform, gauss
from shrimpgrad.engine.runner import realize
from shrimpgrad.future import Thunk
from shrimpgrad.runtime.clang import ClangDevice
from shrimpgrad.util import calc_fan_in_fan_out, calc_gain, prod
import numpy as np

Shape  = Tuple[int, ...]

def pad_left(*shps: Tuple[int, ...], v=1) -> List[Tuple[int ,...]]: return [tuple((v,)*(max(len(s) for s in shps)-len(s)) + s) for s in shps]
def broadcast_shape(*shps: Tuple[int, ...]) -> Tuple[int, ...]: return tuple([max([s[dim] for s in shps]) for dim in range(len(shps[0]))])

class Tensor:
  def __init__(self, shape: Shape, data: Union[List[ConstType], bytes, ConstType, Thunk], dtype:DType=dtypes.float32, device:Device=ClangDevice(), requires_grad:Optional[bool]=None):
    self.requires_grad = requires_grad
    self.grad: Optional[Tensor] = None
    from shrimpgrad.autograd.function import Function
    self.ctx: Optional[Function] = None
    self.cls: Optional[Type[Function]] = None
    if isinstance(data, Thunk): self.thunk = data
    elif isinstance(data, bytes): 
      self.thunk = Thunk.load_from_cpu(np.frombuffer(data, dtype=np.uint8), dtype, shape)
    elif isinstance(data, ConstType): self.thunk = Thunk.load_const(data, shape, dtype, device)
    else:
      if shape == () and not isinstance(data, ConstType) and len(data) == 1: data = data[0]
      self.thunk = Thunk.load_from_cpu(data, dtype, shape)
    if self.thunk.device != device: self.thunk = self.thunk.copy_to_device(device)

  def backward(self) -> Tensor:
    self.grad = Tensor.ones(self.shape, self.dtype, requires_grad=False)
    visited = set()
    topo = []
    # TODO: Turn this into generator so we don't allocate memory for
    # a massive list
    def build_topo(tensor: Tensor):
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

  def nbytes(self): return self.thunk.base.buff.nbytes

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

  def __broadcast(self, y: Union[Tensor, ConstType], reverse=False) -> Tuple[Tensor,...]:
    x = self
    if not isinstance(y, Tensor):
      assert isinstance(y, ConstType), f'type(y)={type(y)} is not a ConstType'
      y = Tensor((), data=y, dtype=dtypes.from_py(y), requires_grad=self.requires_grad)
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
    if self.thunk.base.realized is None: 
      return self.replace(x)
    self.thunk = self.thunk.assign(x.thunk)
    return self
  
  def contiguous(self):
    from shrimpgrad.autograd.function import Contiguous 
    return Contiguous.apply(self)

  def cast(self, dtype: DType) -> Tensor:
    from shrimpgrad.autograd.function import Cast
    if self.dtype == dtype: return self
    return Cast.apply(self, dtype=dtype)

  def clamp(self, min_val=0.0, max_val=1.0) -> Tensor:
    return self.where((self.detach() > min_val), min_val).where(self.detach() < max_val, max_val)

  def where(self, x: Tensor, y: Tensor|ConstType ) -> Tensor:
    from shrimpgrad.autograd.function import Where
    cond, x = self.__broadcast(x)
    cond, y_ = cond.__broadcast(y)
    return Where.apply(cond.cast(dtypes.bool_), *x.__broadcast(y_))

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
  
  def pow(self, n: Tensor|ConstType) -> Tensor:
    if not isinstance(n, Tensor):
      # power of 0
      if n == 0: return 1. + self*0
      # power of 1
      if n == 1: return self
      # power of negative
      if n < 0: return 1. / self.pow(-n)
      # power of int
      if int(n) == n: 
        if n % 2 == 0: return self.square().pow(n//2)
        return self*(self.square().pow((n-1)//2))
    # power of Tensor
    # TODO: Tensor exponent and Float exponent
    return self

  def square(self) -> Tensor: return self * self

  def sqrt(self) -> Tensor:
    from shrimpgrad.autograd.function import Sqrt
    return Sqrt.apply(self)
  
  def rsqrt(self) -> Tensor: return 1.0 / self.sqrt()

  def _canonicalize_axis(self, axis:Optional[int|Tuple[int,...]]):
    axis = axis if axis != None else tuple(i for i in range(self.ndim))
    return tuple(ax if ax >= 0 else ax + self.ndim for ax in (axis if isinstance(axis, Tuple) else (axis,)))

  def _canonicalize_dim(self, dim: int) -> int:
    """
    A dim value within the range [-self.ndim - 1, self.ndim + 1) 
    """
    return dim + self.ndim if dim < 0 else dim

  def mean(self, axis: Optional[int|Tuple[int,...]]=None, keepdim=False) -> Tensor:
    axis_ = self._canonicalize_axis(axis)
    return  self.sum(axis=axis, keepdim=keepdim) / prod([self.shape[i] for i in axis_])

  def sum(self, axis:Optional[Union[int,Tuple[int,...]]]=None, keepdim=False) -> Tensor:
    from shrimpgrad.autograd.function import Sum
    axis_ = self._canonicalize_axis(axis)
    shape = tuple(s for i, s in enumerate(self.shape) if i not in axis_)
    ret = Sum.apply(self, axis=axis_)
    return ret if keepdim else ret.reshape(*shape)
  
  def max(self, axis: Optional[Union[int,Tuple[int,...]]]=None, keepdim=False) -> Tensor:
    from shrimpgrad.autograd.function import Max 
    axis_ = self._canonicalize_axis(axis)
    shape = tuple(s for i, s in enumerate(self.shape) if i not in axis_)
    ret = Max.apply(self, axis=axis_) 
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
  def __mul__(self, other: Tensor|ConstType) -> Tensor: return self.mul(other)
  def __rmul__(self, other): return self.mul(other, reverse=True)
  def __add__(self, other: Tensor|ConstType) -> Tensor: return self.add(other)
  def __radd__(self, other): return self.add(other, reverse=True)
  def __neg__(self): return self.mul(-1.)
  def __sub__(self, other): return self.sub(other)
  def __rsub__(self, other): return self.sub(other, True)
  def __truediv__(self, other): return self.div(other)
  def __rtruediv__(self, other): return self.div(other, reverse=True)
  def __matmul__(self, other) -> Tensor: return self.matmul(other)
  def __pow__(self, n: Tensor|ConstType) -> Tensor: return self.pow(n)
  def __rpow__(self, other): return self.pow(other)

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
  
  def batch_norm(self,  w: Tensor | None, b: Tensor | None,
                 mean: Tensor, var: Tensor, axis: int|Tuple[int, ...]=1):
    axis_ = self._canonicalize_axis(axis)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(self.shape))
    x = self - mean.reshape(*shape)
    if w is not None: x = x * w.reshape(*shape)
    ret = x.mul(var.reshape(*shape) if len(var.shape) == len(axis_) else var)
    return (ret + b.reshape(*shape)) if b is not None else ret
  
  # Statistic Functions
  def std(self, axis: Optional[int|Tuple[int,...]]=None, correction: int=1, keepdim=False) -> Tensor:
    return self.var(axis, correction, keepdim).sqrt()

  def var(self, axis: Optional[int|Tuple[int,...]]=None, correction: int=1, keepdim=False) -> Tensor: 
    squares = (self - self.mean(axis=axis, keepdim=True)).square()
    sum_of_squares = squares.sum(axis=axis, keepdim=True) 
    n = prod([si for si, so in zip(self.shape, sum_of_squares.shape) if si != so])
    if keepdim: return sum_of_squares.div(max(0, n-correction))
    return squares.sum(axis=axis, keepdim=keepdim).div(max(0, n-correction))

  # Loss Functions
  def binary_cross_entropy(self, y: Tensor) -> Tensor: return ((-y)*self.log() - (1.0-y)*((1.0-self).log())).mean()
  def hinge_loss(self, target: Tensor) -> Tensor: return (1.0 + -target*self).relu().sum() * (1.0/target.shape[0])
  def mse(self, target: Tensor) -> Tensor: return (self-target).square().mean()

  # TODO: This is not a correct implementation (placeholder)
  def sparse_categorical_cross_entropy(self, y: Tensor) -> Tensor: return (-y.log()*self).sum() * (1.0/y.shape[0])

  # Shape Shift Functions
  def expand(self, *shps) -> Tensor:
    from shrimpgrad.autograd.function import Expand
    return Expand.apply(self, shape=tuple(shps))

  def reshape(self, *dims) -> Tensor:
    from shrimpgrad.autograd.function import Reshape
    new_shape = tuple([s if s is not None else self.shape[i] for i,s in enumerate(dims)])
    # resolve -1
    if (c := new_shape.count(-1)) > 1: raise RuntimeError(f"only one dimension can be inferred using -1, getting {new_shape}")
    if c: new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape])
    return Reshape.apply(self, shape=new_shape)

  def permute(self, order: Tuple[int,...]) -> Tensor:
    from shrimpgrad.autograd.function import Permute
    return Permute.apply(self, order=order)

  def pad(self, pad_width: Tuple[Tuple[int,int],...], value:ConstType=0.0) -> Tensor:
    from shrimpgrad.autograd.function import Pad 
    return Pad.apply(self, pad_width=pad_width, value=value)

  def shrink(self, shrink_width: Tuple[Tuple[int,int]|None,...]) -> Tensor:
    from shrimpgrad.autograd.function import Shrink 
    if all(x is None or x == (0,s) for x,s in zip(shrink_width, self.shape)): return self
    return Shrink.apply(self, shrink_width=tuple(x if x is not None else (0,s) for x,s in zip(shrink_width, self.shape)))

  def transpose(self, ax0=1, ax1=0):
    ax0, ax1 = (ax0 + self.ndim if ax0 < 0 else ax0), (ax1 + self.ndim if ax1 < 0 else ax1)
    order = [i for i in range(self.ndim)]
    order[ax0], order[ax1] = order[ax1], order[ax0]
    return self.permute(tuple(order))
  
  def flatten(self, start_dim=0, end_dim=-1) -> Tensor:
    end_dim = end_dim if end_dim >=0 else end_dim + self.ndim 
    new_dim = prod(self.shape[start_dim:end_dim+1]) 
    new_shape = tuple([*self.shape[:start_dim], new_dim, *self.shape[end_dim+1:]])
    return self.reshape(*new_shape)

  def squeeze(self, dim:Optional[int]=None) -> Tensor:
    if not dim: return self.reshape(*[si for si in self.shape if si != 1]) 
    dim = self._canonicalize_dim(dim)
    return self if self.shape[dim] != 1 or not self.ndim else self.reshape(*[*self.shape[:dim], *self.shape[dim+1:]])
  
  def unsqueeze(self, dim: int) -> Tensor:
    """
    Returns a new tensor with a dimension of size one inserted at the specified position.

    The returned tensor shares the same underlying data with this tensor.

    A dim value within the range [-input.ndim - 1, input.ndim + 1) can be used. Negative dim will correspond to unsqueeze() applied at dim = dim + input.ndim + 1.
    """
    dim = self._canonicalize_dim(dim)
    return self.reshape(*[*self.shape[:dim], 1, *self.shape[dim:]])

  def tile(self, reps: Tuple[int, ...]) -> Tensor:
    """
    Construct a tensor by repeating self the number of times given by reps.

    If reps has length d, the result will have dimension of max(d, self.ndim).

    If self.ndim < d, self is promoted to be d-dimensional by prepending
    new axes. So a shape (3,) tensor is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the
    desired behavior, promote self to d-dimensions manually before calling this function.

    If self.ndim > d, reps is promoted to self.ndim by prepending 1s to it.
    Thus for self of shape (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2).

    Note: This currently forces a contiguous operation (copy) after expand
    to increase the size of the data buffer.
    """
    local_shape = self.shape
    if len(reps) < len(local_shape): reps = pad_left(reps, local_shape)[0]
    if len(reps) > len(local_shape): local_shape = pad_left(local_shape, reps)[0]
    new_shape, expand_shape = [], []
    for (r, sh) in zip(reps, local_shape):
      if r > 0:
        new_shape.append(1)
        expand_shape.append(r)
      new_shape.append(sh)
      expand_shape.append(sh)
    new_shape = tuple(new_shape)
    expand_shape = tuple(expand_shape)
    y = self.reshape(*new_shape).expand(*expand_shape).contiguous()
    combine_shape = tuple([r*s for r,s in zip(reps, local_shape)])
    return y.reshape(*combine_shape) 
  
  def groupby(self, kernel_shape: Tuple[int, ...], dilation: int|Tuple[int,int]=1, stride: int|Tuple[int,int]=1):
    """
    Given a kernel shape, group the input tensor by the kernel shape based on the
    dilation and stride. Useful in pooling operations such as Conv2D.
    """
    # From https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py (_pool)
    # Assuming 4D input
    # TODO: Simplify, change, ndim support
    iH, iW = self.shape[-2:]
    kH, kW = kernel_shape[-2:]
    if isinstance(stride, Tuple): sH, sW = stride
    else: sH, sW = stride, stride
    if isinstance(dilation, Tuple): dH, dW = dilation
    else: dH, dW = dilation, dilation
    oH = math.ceil((iH - (dH*(kH-1))) / sH)
    oW = math.ceil((iW - (dW*(kW-1))) / sW)
    noop = [None]*len(self.shape[:-2])
    # Expand maximally to cover a kernel for each dim and each dilation
    y = self.tile(tuple([1]*len(noop)+ [math.ceil(kH*(iH+dH)/iH), math.ceil(kW*(iW+dW)/iW)]))
    y = y.shrink((*noop, (0,kH*(iH+dH)),(0,kW*(iW+dW)))).contiguous().reshape(*noop, kH, iH+dH, kW, iW+dW)
    y = y.shrink((*noop, (0,kH), (0,oH*sH), (0,kW),(0,oW*sW))).reshape(*noop, *(kH,oH,sH, kW,oW, sW))
    y = y.shrink((*noop, (0,kH), (0,oH), (0,1), (0,kW),(0,oW), (0,1))).reshape(*noop, kH,oH, kW, oW)
    return y.permute(tuple([i for i in range(len(noop))] + [len(noop)+i*2+1 for i in range(2)]  + [len(noop)+i*2 for i in range(2)]))

  # Neural Network Layer Functions
  def linear(self, w: Tensor, bias:Optional[Tensor]=None) -> Tensor:
    return self.dot(w) + bias if bias else self.dot(w)

  def conv2d(self, w: Tensor, bias: Optional[Tensor]=None,
             stride:int|Tuple[int,int]=1, 
             padding:int|Tuple[Tuple[int,int],...]=0,
             dilation:int|Tuple[int,int]=1, groups:int=1) -> Tensor: 
    """
    self.shape: (minibatch,in_channels,iH,iW)
    kernel.shape: (out_channels, in_channels/groups, kH, kW)
    bias.shape: (out_channels,) or None
    stride: int or tuple (sH, sW)
    padding: int or tuple of tuples for each dimension
    dilation: int or (dH, dW)
    groups: int (in_channels and out_channels must be divisible by groups)
    """
    # From https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py (conv2s normal conv)
    assert self.ndim == 4, "conv2d supports only 4D shapes for now"
    assert w.shape[1] == self.shape[-3]//groups, f"kernel shape {w.shape} dim 1 is invalid"
    assert bias is None or bias.shape == (w.shape[0], ), f"bias shape is invalid {bias.shape = }"
    B = self.shape[0]
    kH, kW = w.shape[-2:]
    HW = (kH,kW)
    cO = w.shape[0] 
    cI = self.shape[1]
    if isinstance(padding, int):
      padding = tuple([(0,0),(0,0)]+[(padding,padding) for _ in range(self.ndim-2)])
    x = self.pad(padding).contiguous().groupby((kH,kW), dilation, stride)
    rcout, oyx = cO // groups, x.shape[2:-2]
    x = x.reshape(B, groups, cI, 1, *oyx, *HW).expand(B, groups, cI, rcout, *oyx, *HW).permute((0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))]))
    ret = (x * w.reshape(1, groups, rcout, *[1] * len(oyx), cI, *HW)).sum(tuple([-1-i for i in range(1+len(oyx))]), keepdim=True).reshape(B, cO, *oyx)
    return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))
  
  def maxpool2d(self, kernel_size: int|Tuple[int,...], stride: int|Tuple[int,int]=1, padding: int|Tuple[Tuple[int,int],...]=0, dilation:int|Tuple[int,int]=1) -> Tensor:
    assert self.ndim == 4, "conv2d supports only 4D shapes for now"
    kH, kW = kernel_size if isinstance(kernel_size,tuple) else (kernel_size, kernel_size) 
    if isinstance(padding, int):
      padding = tuple([(0,0),(0,0)]+[(padding,padding) for _ in range(self.ndim-2)])
    x = self.pad(padding).contiguous().groupby((kH,kW), dilation, stride)
    return x.max((-2,-1), keepdim=True).reshape(*x.shape[:-2])
    
  def sequential(self, ll:List[Callable[[Tensor], Tensor]]): return functools.reduce(lambda x,f: f(x), ll, self)

  # Creation Functions
  @staticmethod
  def zeros(shape: Shape, dtype:DType=dtypes.float32, **kwargs) -> Tensor: return Tensor.full(shape, fill_value=0.0, dtype=dtype, **kwargs)

  @staticmethod
  def ones(shape: Shape, dtype:DType=dtypes.float32, **kwargs) -> Tensor:
    return Tensor.full(shape, fill_value=dtypes.cast(dtype, 1.0), dtype=dtype, **kwargs)

  @staticmethod
  def arange(start: int, stop:int, step=1, dtype:DType=dtypes.float32, **kwargs) -> Tensor: return Tensor((math.ceil((stop - start) / step),), [float(i) if dtype == dtypes.float32 else int(i) for i in range(start, stop, step)], dtype, **kwargs)

  @staticmethod
  def fromlist(shape: Shape, data:List[ConstType], dtype=dtypes.float32, **kwargs): return Tensor(shape, data=data, dtype=dtype, **kwargs)

  @staticmethod
  def frombytes(shape: Shape, data: bytes, dtype=dtypes.float32, **kwargs): return Tensor(shape, data, dtype)

  @staticmethod
  def full(shape: Shape, fill_value: ConstType, dtype=dtypes.float32, **kwargs) -> Tensor:
    if not len(shape): return Tensor((), fill_value)
    return Tensor(shape, fill_value, **kwargs)

  @staticmethod
  def full_like(t: Tensor, fill_value: ConstType, **kwargs) -> Tensor: return Tensor.full(t.shape, fill_value=fill_value, dtype=t.dtype, **kwargs)

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

  def const(self, val:ConstType, **kwargs) -> Tensor: return Tensor.full_like(self, val, **kwargs)

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
  def data(self) -> np.ndarray:
    # TODO: Change this to something worthy
    base = self.thunk.base
    if hasattr(base, 'buff'):
      data = base.buff.pointer(to_ctype(self.dtype))
      if not self.thunk.vt.contiguous or self.thunk.base.buff.size > self.numel:
        strides = tuple([s*self.dtype.bytes for s in self.thunk.vt.strides])
        arr = np.frombuffer(data, dtype=to_numpy(self.dtype))
        arr = np.lib.stride_tricks.as_strided(arr, shape=self.shape, strides=strides)
        return arr
      if base.shape == ():
        return np.frombuffer(data, dtype=to_numpy(self.dtype)).reshape(())
    else:
      raise TypeError("self is not realized where is the buff")
    return np.frombuffer(data, dtype=to_numpy(self.dtype)).reshape(self.shape)

  def numpy(self) -> np.ndarray: return self.realize().data()

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

  def __repr__(self): return f"<Tensor {self.thunk!r} on {self.device} with grad {(self.grad.thunk if self.grad is not None else None)!r} {self.dtype} {self.shape}>"
  def __str__(self): return self.__repr__()
  def __hash__(self): return id(self)