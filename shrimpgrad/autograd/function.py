import math
from typing import Any, Optional, Tuple, TypeAlias 
import shrimpgrad as shrimp
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from shrimpgrad.util import argsort
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
    return x.alu(BinaryOps.MAX, FutureTensor.zeros_like(x))
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    x = ctx.saved_tensors[0]
    # dx' = 0 if x < else 1
    dx = FutureTensor.zeros_like(x).alu(BinaryOps.CMPLT, x)
    return dx.alu(BinaryOps.MUL, grad_out) 

class Sum(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, axis: Tuple[int,...]=(0,), keepdim=False) -> FutureTensor:
    ctx.save_for_backward(x)
    return x.reduce(ReduceOps.SUM, ax=axis, keepdim=keepdim) 
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
    return x.alu(UnaryOps.LOG2).alu(BinaryOps.MUL, x.const(math.log(2)))
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    # dL/dx = 1/x * grad_out
    x = ctx.saved_tensors[0]
    return grad_out.alu(BinaryOps.DIV, x)

class Reshape(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, shape: Tuple[int,...] ) -> FutureTensor:
    ctx.save_for_backward(x)
    return x.reshape(shape)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    x = ctx.saved_tensors[0] 
    return grad_out.reshape(*x.shape)

class Permute(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, order: Tuple[int, ...]) -> FutureTensor:
    ctx.save_for_backward(x)
    ctx.order = order
    return x.permute(order) 
  @staticmethod 
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    return grad_out.permute(argsort(ctx.order))
  
class Expand(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor, shape: Tuple[int, ...]) -> FutureTensor:
    ctx.save_for_backward(x)
    ctx.expanded_axis = []
    return x.expand(shape) 
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    return grad_out.sum(axis=tuple(ctx.expanded_axis)) 

class Less(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:FutureTensor, y:FutureTensor) -> FutureTensor: 
    return x.alu(BinaryOps.CMPLT, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:FutureTensor) -> OptionalGradients: return None, None

class Eq(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:FutureTensor, y:FutureTensor) -> FutureTensor:
    return x.alu(BinaryOps.CMPEQ, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:FutureTensor) -> OptionalGradients: return None, None

class Cast(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x:FutureTensor, dtype: shrimp.DType) -> FutureTensor:
    ctx.save_for_backward(x)
    ctx.input_dtype = x.dtype
    return x 
  @staticmethod
  def backward(ctx: FunctionContext, grad_out:FutureTensor) -> FutureTensor:
    # grad_out.data = map(functools.partial(dtypes.cast, ctx.input_dtype), grad_out.data)
    return grad_out.cast(ctx.input_dtype)

class Where(Function):
  @staticmethod
  def forward(ctx: FunctionContext, condition: FutureTensor, x: FutureTensor, y: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x, y)
    ctx.cond = condition
    return condition.alu(TernaryOps.WHERE, x, y)
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> OptionalGradients:
    cond = ctx.cond
    return cond.alu(TernaryOps.WHERE, grad_out, grad_out.const(0.0)), \
      cond.alu(TernaryOps.WHERE, grad_out.const(0.0), grad_out) 

class Sigmoid(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: FutureTensor) -> FutureTensor:
    ctx.save_for_backward(x)
    a = x.alu(BinaryOps.MUL, x.const(-1.0/math.log(2))).alu(UnaryOps.EXP2)
    b = a.const((1.0)).alu(BinaryOps.ADD, a)
    c = b.const(1.0).alu(BinaryOps.ADD, b)
    ctx.ret = c.const(1.0).alu(BinaryOps.DIV, c) 
    return ctx.ret
  @staticmethod
  def backward(ctx: FunctionContext, grad_out: FutureTensor) -> FutureTensor:
    return ctx.ret.alu(BinaryOps.MUL, ctx.ret.const(1).alu(BinaryOps.SUB, ctx.ret)).alu(BinaryOps.MUL, grad_out)
