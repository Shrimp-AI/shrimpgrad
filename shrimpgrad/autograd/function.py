import math
from typing import Any, Tuple 
import shrimpgrad as shrimp
from shrimpgrad.runtime.python import PythonRuntime, BinaryOps, ReduceOps, UnaryOps
from shrimpgrad.util import flatten

class FunctionContext:
  # TODO: Implement device abstraction
  def __init__(self, device: str): 
    self.device = device

  def save_for_backward(self, *tensors: shrimp.Tensor):
    self.saved_tensors = tuple(tensors)
  
class Function(FunctionContext):
  @staticmethod
  def forward(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
      "You must implement the forward function for autograd.Function."
    )

  @staticmethod
  def backward(ctx: Any, *grad_outputs: Any) -> Any:
    raise NotImplementedError(
      "You must implement the backward function for autograd.Function. "
    )

  @classmethod
  def apply(cls, *tensors, **kwargs) -> shrimp.Tensor:
    ctx = cls(tensors[0].device)
    ret = cls.forward(ctx, *tensors, **kwargs)
    ret.cls = cls
    ret.ctx = ctx
    return ret

class Add(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor, y: shrimp.Tensor) -> shrimp.Tensor:
    ctx.save_for_backward(x, y)
    return PythonRuntime.exec(BinaryOps.ADD, x, y)

  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor) -> shrimp.Tensor:
    return (grad_out, grad_out)

class Mul(Function):
  @staticmethod
  def forward(ctx:FunctionContext, x: shrimp.Tensor, y: shrimp.Tensor) -> shrimp.Tensor:
    ctx.save_for_backward(x, y)
    return PythonRuntime.exec(BinaryOps.MUL, x, y)

  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor) -> shrimp.Tensor:
    x, y = ctx.saved_tensors
    return (PythonRuntime.exec(BinaryOps.MUL, y, grad_out), PythonRuntime.exec(BinaryOps.MUL, x, grad_out))

class Div(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor, y: shrimp.Tensor) -> shrimp.Tensor:
    ctx.save_for_backward(x, y)
    return PythonRuntime.exec(BinaryOps.DIV, x, y)

  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor) -> shrimp.Tensor:
    x, y = ctx.saved_tensors
    '''
    dz/dx -> x/y = x * y^-1 = 1/y
    dz/dy -> x/y = x * y^-1 = -x*y^-2 = -x/y^2
    
    ''' 
    numerator = PythonRuntime.exec(UnaryOps.NEG, x)
    numerator = PythonRuntime.exec(BinaryOps.MUL, numerator, grad_out)
    denominator = PythonRuntime.exec(BinaryOps.MUL, y ,y)
    dzdy = PythonRuntime.exec(BinaryOps.DIV, numerator, denominator)
    return PythonRuntime.exec(BinaryOps.DIV, grad_out, y), dzdy

class Exp(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor) -> shrimp.Tensor:
    ctx.save_for_backward(x)
    ctx.ret = PythonRuntime.exec(UnaryOps.EXP2, PythonRuntime.exec(BinaryOps.MUL, x, x.const(1/math.log(2))))
    return ctx.ret

  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor) -> shrimp.Tensor:
    # f(x) = e^x then dy/dx = e^x
    return PythonRuntime.exec(BinaryOps.MUL, ctx.ret, grad_out)

class ReLU(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor) -> shrimp.Tensor:
    ctx.save_for_backward(x)
    return PythonRuntime.exec(BinaryOps.MAX, x, shrimp.Tensor.zeros_like(x))

  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor):
    x = ctx.saved_tensors[0]
    # dx' = 0 if x < else 1
    dx = PythonRuntime.exec(BinaryOps.CMPLT, shrimp.Tensor.zeros_like(x), x)
    return PythonRuntime.exec(BinaryOps.MUL, dx, grad_out) 

class Sum(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor, axis: Tuple[int,...]=(0,), keepdim=False) -> shrimp.Tensor:
    ctx.save_for_backward(x)
    return PythonRuntime.exec(ReduceOps.SUM, x, ax=axis, keepdim=keepdim) 
    
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor):
    x = ctx.saved_tensors[0]
    return grad_out.expand(*x.shape)

class Log(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor, axis: Tuple[int,...]=(0,)) -> shrimp.Tensor:
    ctx.save_for_backward(x)
    # ln(x) = log2(x)/log2(e) <by change of base>
    #       = log2(x) / (1/ln(2)) = log2(x) * ln(2)
    return PythonRuntime.exec(BinaryOps.MUL, PythonRuntime.exec(UnaryOps.LOG2, x), x.const(math.log(2)))

  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor):
    # dL/dx = 1/x * grad_out
    x = ctx.saved_tensors[0]
    return PythonRuntime.exec(BinaryOps.DIV, grad_out, x)

class Reshape(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor, shape: Tuple[int,...] ) -> shrimp.Tensor:
    ctx.save_for_backward(x)
    if shrimp.util.prod(shape) != x.numel: raise RuntimeError(f'shape \'{shape}\' is invalid for input of size {x.numel}')
    if x.contiguous:
      if shape == x.shape:
        return shrimp.Tensor(shape, x.data, dtype=x.dtype)
      # Reshape from scalar to n-dim tensor 
      if x.is_scalar() and len(shape):
        return shrimp.Tensor(shape, [x.data], dtype=x.dtype)
      return shrimp.Tensor(shape, x.data if len(shape) else x.data[0], dtype=x.dtype)
    return shrimp.Tensor(shape, flatten(x), dtype=x.dtype)

  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor):
    x = ctx.saved_tensors[0] 
    return grad_out.reshape(*x.shape)

class Permute(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor, order: Tuple[int, ...]) -> shrimp.Tensor:
    ctx.save_for_backward(x)
    ctx.order = order
    new_shape = [x.shape[i] for i in order]
    new_strides = [x.strides[i] for i in order]
    out = shrimp.Tensor(tuple(new_shape), x.data, dtype=x.dtype) 
    out.strides = new_strides
    out.contiguous = False
    return out 

  @staticmethod 
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor):
    return grad_out.permute(shrimp.util.argsort(ctx.order))
  
class Expand(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor, shape: Tuple[int, ...]) -> shrimp.Tensor:
    ctx.save_for_backward(x)
    ctx.expanded_axis = []
    out = shrimp.Tensor.zeros_like(x)
    for i, (si, so) in enumerate(zip(x.shape, shape)):
      if si != so: 
        out.strides[i] = 0
        ctx.expanded_axis.append(i)
    out.shape = shape
    out.data = x.data
    return out
 
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: shrimp.Tensor) -> shrimp.Tensor:
    return grad_out.sum(axis=tuple(ctx.expanded_axis)) 
