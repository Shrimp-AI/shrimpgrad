from __future__ import annotations
import math
from typing import Any, Optional, Tuple, Type, TypeAlias
from shrimpgrad.device import Device
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from shrimpgrad.tensor import Tensor
from shrimpgrad.dtype import ConstType, DType
from shrimpgrad.util import argsort
from shrimpgrad.future import Thunk

OptionalGradients: TypeAlias = Tuple[Optional[Thunk], ...]

class FunctionContext:
  def __init__(self, device: Device, *tensors):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    self.tensors = tensors
  def __setattr__(self, name, value): self.__dict__[name] = value
  def save(self, name, value): self.__setattr__(name, value)
  def load(self, name): return self.__dict__[name]

class Function(FunctionContext):
  @staticmethod
  def forward(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
      "You must implement the forward function for autograd.Function."
    )

  @staticmethod
  def backward(ctx: FunctionContext, *grad_outputs: Any) -> Any:
    raise NotImplementedError(
      "You must implement the backward function for autograd.Function. "
    )

  @classmethod
  def apply(cls: Type[Function], *tensors, **kwargs) -> Tensor:
    ctx = cls(tensors[0].device, *tensors)
    thunk = cls.forward(ctx, *[t.thunk for t in tensors], **kwargs)
    ret = Tensor.__new__(Tensor)
    ret.grad, ret.requires_grad, ret.cls, ret.ctx = None, ctx.requires_grad, cls, ctx if ctx.requires_grad else None
    ret.thunk= thunk
    return ret

class Add(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, y: Thunk) -> Thunk:
    return x.alu(BinaryOps.ADD, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> OptionalGradients:
    return grad_out if ctx.needs_input_grad[0] else None, grad_out if ctx.needs_input_grad[1] else None

class Sub(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:Thunk, y:Thunk) -> Thunk:
    return x.alu(BinaryOps.SUB, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_output:Thunk) -> OptionalGradients:
    return grad_output if ctx.needs_input_grad[0] else None, \
           grad_output.alu(UnaryOps.NEG) if ctx.needs_input_grad[1] else None

class Mul(Function):
  @staticmethod
  def forward(ctx:FunctionContext, x: Thunk, y: Thunk) -> Thunk:
    ctx.save('x', x)
    ctx.save('y', y)
    return x.alu(BinaryOps.MUL, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> OptionalGradients:
    x, y = ctx.load('x'), ctx.load('y')
    return y.alu(BinaryOps.MUL, grad_out) if ctx.needs_input_grad[0] else None, \
      x.alu(BinaryOps.MUL, grad_out) if ctx.needs_input_grad[1] else None

class Div(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, y: Thunk) -> Thunk:
    ctx.save('x', x)
    ctx.save('y', y)
    return x.alu(BinaryOps.DIV, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> OptionalGradients:
    '''
    dz/dx -> x/y = x * y^-1 = 1/y
    dz/dy -> x/y = x * y^-1 = -x*y^-2 = -x/y^2
    '''
    x, y = ctx.load('x'), ctx.load('y')
    numerator = x.alu(UnaryOps.NEG)
    numerator = numerator.alu(BinaryOps.MUL, grad_out)
    denominator = y.alu(BinaryOps.MUL, y)
    dzdy = numerator.alu(BinaryOps.DIV, denominator)
    return grad_out.alu(BinaryOps.DIV, y) if ctx.needs_input_grad[0] else None, dzdy if ctx.needs_input_grad[1] else None

class Exp(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk) -> Thunk:
    ctx.save('ret', ret:=x.alu(BinaryOps.MUL, x.const(1/math.log(2))).alu(UnaryOps.EXP2))
    return ret
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    # f(x) = e^x then dy/dx = e^x
    return ctx.load('ret').alu(BinaryOps.MUL, grad_out)

class ReLU(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk) -> Thunk:
    ctx.save('x', x) 
    return x.alu(BinaryOps.MAX, x.const(0.0))
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    x = ctx.load('x')
    # dx' = 0 if x < else 1
    dx = x.const(0.0).alu(BinaryOps.CMPLT, x)
    return dx.alu(BinaryOps.MUL, grad_out)

class Sum(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, axis: Tuple[int,...]=(0,)) -> Thunk:
    ctx.save('x', x)
    return x.reduce(ReduceOps.SUM, axis=axis)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    x = ctx.load('x')
    return grad_out.expand(x.shape)

class Max(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, axis: Tuple[int,...]=(0,)) -> Thunk:
    ctx.save('x', x)
    ctx.save('axis', axis)
    ret = x.reduce(ReduceOps.MAX, axis=axis)
    ctx.save('ret', ret)
    return ret
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    x: Thunk = ctx.load('x')
    ret: Thunk = ctx.load('ret')
    axis: Tuple[int, ...] = ctx.load('axis')
    # dL/dx = grad_out * [0 if not max, 1 if max]; max can occur multiple times in a Tensor
    # spread over all locations where max
    ret = ret.expand(x.shape)
    dx = ret.alu(BinaryOps.CMPEQ, x).alu(TernaryOps.WHERE, ret.const(1.), ret.const(0.))
    cnt = dx.reduce(ReduceOps.SUM, axis)
    dx = dx.alu(BinaryOps.DIV, cnt)
    return grad_out.expand(x.shape).alu(BinaryOps.MUL, dx)

class Log(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, axis: Tuple[int,...]=(0,)) -> Thunk:
    # ln(x) = log2(x)/log2(e) <by change of base>
    #       = log2(x) / (1/ln(2)) = log2(x) * ln(2)
    ctx.save('x', x)
    return x.alu(UnaryOps.LOG2).alu(BinaryOps.MUL, x.const(math.log(2)))
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    # dL/dx = 1/x * grad_out
    x = ctx.load('x')
    return grad_out.alu(BinaryOps.DIV, x)

class Reshape(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, shape: Tuple[int,...] ) -> Thunk:
    ctx.save('x', x)
    return x.reshape(shape)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    x = ctx.load('x')
    return grad_out.reshape(x.shape)

class Permute(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, order: Tuple[int, ...]) -> Thunk:
    ctx.save('order' ,order)
    return x.permute(order)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    return grad_out.permute(argsort(ctx.load('order')))

class Expand(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, shape: Tuple[int, ...]) -> Thunk:
    ctx.save('expanded_axis',[i for i,(si,so) in enumerate(zip(x.shape, shape)) if si != so])
    return x.expand(shape)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    return grad_out.reduce(ReduceOps.SUM, axis=tuple(ctx.load('expanded_axis')))

class Pad(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, pad_width: Tuple[Tuple[int,int], ...], value: ConstType) -> Thunk:
    ctx.save('arg', tuple([(pw[0], pw[0]+s) for pw,s in zip(pad_width, x.shape)]))
    return x.pad(pad_width, value)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    return grad_out.shrink(ctx.load('arg'))

class Shrink(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk, shrink_width: Tuple[Tuple[int,int], ...]) -> Thunk:
    ctx.save('arg', tuple([(sw[0], s - sw[1]) for sw,s in zip(shrink_width, x.shape)]))
    return x.shrink(shrink_width)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    return grad_out.pad(ctx.load('arg'))

class Less(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:Thunk, y:Thunk) -> Thunk:
    return x.alu(BinaryOps.CMPLT, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:Thunk) -> OptionalGradients: return None, None

class Eq(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:Thunk, y:Thunk) -> Thunk:
    return x.alu(BinaryOps.CMPEQ, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:Thunk) -> OptionalGradients: return None, None

class Cast(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:Thunk, dtype: DType) -> Thunk:
    ctx.save('input_dtype', x.dtype)
    return x.cast(dtype)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:Thunk) -> Thunk:
    return grad_out.cast(ctx.load('input_dtype'))

class Where(Function):
  @staticmethod
  def forward(ctx: FunctionContext, condition: Thunk, x: Thunk, y: Thunk) -> Thunk:
    ctx.save('cond', condition)
    return condition.alu(TernaryOps.WHERE, x, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> OptionalGradients:
    cond = ctx.load('cond')
    return None, cond.alu(TernaryOps.WHERE, grad_out, grad_out.const(0.0)) if ctx.needs_input_grad[1] else None, \
      cond.alu(TernaryOps.WHERE, grad_out.const(0.0), grad_out) if ctx.needs_input_grad[2] else None

class Sigmoid(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: Thunk) -> Thunk:
    a = x.alu(BinaryOps.MUL, x.const(-1.0/math.log(2))).alu(UnaryOps.EXP2)
    b = a.const(1.0).alu(BinaryOps.ADD, a)
    ctx.save('ret', ret:=b.const(1.0).alu(BinaryOps.DIV, b))
    return ret 
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: Thunk) -> Thunk:
    ret = ctx.load('ret')
    return ret.alu(BinaryOps.MUL, ret.const(1).alu(BinaryOps.SUB, ret)).alu(BinaryOps.MUL, grad_out)

class Contiguous(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:Thunk) -> Thunk:
    return x.contiguous()
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:Thunk) -> Thunk: return grad_out 
