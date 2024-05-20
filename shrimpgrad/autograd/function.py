import functools
import math
from typing import Any, Optional, Tuple, TypeAlias 
import shrimpgrad as shrimp
from shrimpgrad.dtype import dtypes
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from shrimpgrad.util import flatten, prod, argsort
from shrimpgrad.future import FutureTensor

OptionalGradients: TypeAlias = Tuple[Optional[FutureTensor], ...]

class FunctionContext:
  # TODO: Implement device abstraction
  def __init__(self, device: str): 
    self.device = device

  def save_for_backward(self, *tensors: FutureTensor):
    self.saved_tensors = tuple(tensors)
  
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
  def apply(cls, *tensors, **kwargs) -> FutureTensor:
    ctx = cls(tensors[0].device)
    ret = cls.forward(ctx, *tensors, **kwargs)
    ret.cls = cls
    ret.ctx = ctx
    return ret

class Add(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, y: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x, y)
    return x.alu(BinaryOps.ADD, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> OptionalGradients:
    return (grad_out, grad_out)

class Sub(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:FutureTensor, y:FutureTensor) -> FutureTensor: 
    ctx.save_for_backward(x,y)
    return x.alu(BinaryOps.SUB, y) 
  @staticmethod
  def backward(ctx: FunctionContext, grad_output:FutureTensor) -> OptionalGradients: 
    return grad_output, \
           grad_output.alu(UnaryOps.NEG) 

class Mul(Function):
  @staticmethod
  def forward(ctx:FunctionContext, x: FutureTensor, y: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x, y)
    return x.alu(BinaryOps.MUL, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> OptionalGradients:
    x, y = ctx.saved_tensors
    return (x.alu(BinaryOps.MUL, grad_out), y.alu(BinaryOps.MUL, grad_out))

class Div(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, y: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x, y)
    return x.alu(BinaryOps.DIV, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> OptionalGradients:
    x, y = ctx.saved_tensors
    '''
    dz/dx -> x/y = x * y^-1 = 1/y
    dz/dy -> x/y = x * y^-1 = -x*y^-2 = -x/y^2
    ''' 
    numerator = x.alu(UnaryOps.NEG)
    numerator = numerator.alu(BinaryOps.MUL, grad_out)
    denominator = y.alu(BinaryOps.MUL, y)
    dzdy = numerator.alu(BinaryOps.DIV, denominator)
    return grad_out.alu(BinaryOps.DIV, y), dzdy

class Exp(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x)
    ctx.ret =  x.alu(BinaryOps.MUL, x.const(1/math.log(2))).alu(UnaryOps.EXP2)
    return ctx.ret
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    # f(x) = e^x then dy/dx = e^x
    return ctx.ret.alu(BinaryOps.MUL, grad_out)

class ReLU(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x)
    return PythonRuntime.exec(BinaryOps.MAX, x, FutureTensor.zeros_like(x))
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    x = ctx.saved_tensors[0]
    # dx' = 0 if x < else 1
    dx = PythonRuntime.exec(BinaryOps.CMPLT, FutureTensor.zeros_like(x), x)
    return PythonRuntime.exec(BinaryOps.MUL, dx, grad_out) 

class Sum(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, axis: Tuple[int,...]=(0,), keepdim=False) -> FutureTensor:
    ctx.save_for_backward(x)
    return PythonRuntime.exec(ReduceOps.SUM, x, ax=axis, keepdim=keepdim) 
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    x = ctx.saved_tensors[0]
    return grad_out.expand(*x.shape)

class Log(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, axis: Tuple[int,...]=(0,)) -> FutureTensor:
    ctx.save_for_backward(x)
    # ln(x) = log2(x)/log2(e) <by change of base>
    #       = log2(x) / (1/ln(2)) = log2(x) * ln(2)
    return PythonRuntime.exec(BinaryOps.MUL, PythonRuntime.exec(UnaryOps.LOG2, x), x.const(math.log(2)))
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    # dL/dx = 1/x * grad_out
    x = ctx.saved_tensors[0]
    return PythonRuntime.exec(BinaryOps.DIV, grad_out, x)

class Reshape(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, shape: Tuple[int,...] ) -> FutureTensor:
    ctx.save_for_backward(x)
    if prod(shape) != x.numel: raise RuntimeError(f'shape \'{shape}\' is invalid for input of size {x.numel}')
    if x.contiguous:
      if shape == x.shape:
        return FutureTensor(shape, x.data, dtype=x.dtype)
      # Reshape from scalar to n-dim tensor 
      if x.is_scalar() and len(shape):
        return FutureTensor(shape, [x.data], dtype=x.dtype)
      return FutureTensor(shape, x.data if len(shape) else x.data[0], dtype=x.dtype)
    return FutureTensor(shape, flatten(x), dtype=x.dtype)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    x = ctx.saved_tensors[0] 
    return grad_out.reshape(*x.shape)

class Permute(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, order: Tuple[int, ...]) -> FutureTensor:
    ctx.save_for_backward(x)
    ctx.order = order
    new_shape = [x.shape[i] for i in order]
    new_strides = [x.strides[i] for i in order]
    out = FutureTensor(tuple(new_shape), x.data, dtype=x.dtype) 
    out.strides = new_strides
    out.contiguous = False
    return out 
  @staticmethod 
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    return grad_out.permute(argsort(ctx.order))
  
class Expand(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, shape: Tuple[int, ...]) -> FutureTensor:
    ctx.save_for_backward(x)
    ctx.expanded_axis = []
    out = FutureTensor.zeros_like(x)
    for i, (si, so) in enumerate(zip(x.shape, shape)):
      if si != so: 
        out.strides[i] = 0
        ctx.expanded_axis.append(i)
    out.shape = shape
    out.data = x.data
    return out
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    return grad_out.sum(axis=tuple(ctx.expanded_axis)) 

class Less(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:FutureTensor, y:FutureTensor) -> FutureTensor: 
    return PythonRuntime.exec(BinaryOps.CMPLT, x, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:FutureTensor) -> OptionalGradients: return None, None

class Eq(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:FutureTensor, y:FutureTensor) -> FutureTensor:
    return PythonRuntime.exec(BinaryOps.CMPEQ, x, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:FutureTensor) -> OptionalGradients: return None, None

class Cast(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:FutureTensor, dtype: shrimp.DType) -> FutureTensor:
    ctx.save_for_backward(x)
    ctx.input_dtype = x.dtype
    x.data = list(map(functools.partial(dtypes.cast, dtype), x.data)) if not x.is_scalar() else dtypes.cast(dtype, x.data)
    return x 
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:FutureTensor) -> FutureTensor:
    grad_out.data = map(functools.partial(dtypes.cast, ctx.input_dtype), grad_out.data)
    return grad_out

class Where(Function):
  @staticmethod
  def forward(ctx: FunctionContext, condition: FutureTensor, x: FutureTensor, y: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x, y)
    ctx.cond = condition
    return PythonRuntime.exec(TernaryOps.WHERE, condition, x, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> OptionalGradients:
    cond = ctx.cond
    return PythonRuntime.exec(TernaryOps.WHERE, cond, grad_out, grad_out.const(0.0)), \
      PythonRuntime.exec(TernaryOps.WHERE, cond, grad_out.const(0.0), grad_out) 

class Sigmoid(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x)
    x_ = PythonRuntime.exec(BinaryOps.MUL, x, x.const(-1.0/math.log(2)))
    x_ = PythonRuntime.exec(UnaryOps.EXP2, x_)
    x_ = PythonRuntime.exec(BinaryOps.ADD, x_.const(1.0), x_)
    ctx.ret = PythonRuntime.exec(BinaryOps.DIV, x_.const(1.0), x_)
    return ctx.ret
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    return PythonRuntime.exec(BinaryOps.MUL, PythonRuntime.exec(BinaryOps.MUL, ctx.ret, PythonRuntime.exec(BinaryOps.SUB, ctx.ret.const(1.0), ctx.ret)), grad_out)
