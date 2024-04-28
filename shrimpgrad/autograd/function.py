from typing import Any
import shrimpgrad as shrimp
from shrimpgrad.runtime.python import PythonRuntime, BinaryOps, UnaryOps, ReduceOps


class FunctionContext:
  # TODO: Implement device abstraction
  def __init__(self, device: str): 
    self.device = device

  def save_for_backward(self, *tensors: shrimp.Tensor):
    self.saved_tensors= tuple(tensors)
  
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
  def apply(cls, *tensors, **kwargs):
    ctx = cls(tensors[0].device)
    ret = cls.forward(ctx, *tensors, **kwargs)
    ret.ctx = ctx
    return ret

class Add(Function):
  @staticmethod
  def forward(ctx: FunctionContext, x: shrimp.Tensor, y: shrimp.Tensor) -> shrimp.Tensor:
    ctx.save_for_backward(x, y)
    return shrimp.Tensor(x.shape, PythonRuntime.exec(BinaryOps.Add, x, y), dtype=x.dtype)

  @staticmethod
  def backward(ctx: FunctionContext, grad_out):
    x, y = ctx.saved_tensors
    x.grad += grad_out 
    y.grad += grad_out 

class Mul(Function):
  @staticmethod
  def forward(ctx:FunctionContext, x: shrimp.shrimp.Tensor, y: shrimp.Tensor) -> shrimp.Tensor:
    ctx.save_for_backward(x, y)
    return shrimp.Tensor(x.shape, PythonRuntime.exec(BinaryOps.Mul, x, y), dtype=x.dtype)

  @staticmethod
  def backward(ctx: FunctionContext, grad_out):
    x, y = ctx.saved_tensors
    x.grad += y * grad_out 
    y.grad += x * grad_out 

class ReLU(Function):
  def forward(self, a: shrimp.Tensor) -> shrimp.Tensor:
    if a.is_scalar():
      self.out = shrimp.Tensor((), (0 if a.data < 0 else a.data))
      return self.out

    result = []
    unary_op(lambda x: 0.0 if x < 0 else x, a, 0, 0, a.calc_loops(None), result)
    self.out = shrimp.Tensor(a.shape, result, dtype=a.dtype)
    return self.out
  
  def backward(self):
    a = self.parents[0]
    if a.is_scalar():
      a.grad += (a.data > 0) * self.out.grad
      return

    result = []
    unary_op(lambda x: 1 if x > 0 else 0, a,  0, 0, a.calc_loops(None), result)
    x = shrimp.Tensor(a.shape, result, dtype=a.dtype, requires_grad=False) 
    a.grad += x * a * self.out.grad

class Pow(Function):
  def forward(self, a:shrimp.Tensor, b: shrimp.Tensor) -> shrimp.Tensor:
    if a.is_scalar():
      self.out =  shrimp.Tensor((), a.data ** b.data)
      return self.out
    
    result = []
    binary_op(operator.pow, a, b, 0, 0, 0, a.calc_loops(None), result)
    self.out = shrimp.Tensor(a.shape, result, dtype=a.dtype)

    return self.out
  
  def backward(self):
    a, b = self.parents[0], self.parents[1]
    if a.is_scalar():
      a.grad += b * (a ** (b - 1)) * self.out.grad
    else:
      a.grad += b * (a ** (b - shrimp.Tensor((1,), [1]))) * self.out.grad

class Sum(Function):
  def forward(self, x: shrimp.Tensor, axis: int=0, keepdim=False) -> shrimp.Tensor:
    ret = reduce_(operator.add, x, x.calc_loops(None), 0, 0, ax=axis)
    self.out = shrimp.Tensor((*x.shape[0:axis],*[1]*(keepdim), *x.shape[axis+1:]), ret)
    return self.out

  def backward(self):
    x = self.parents[0]
    x.grad = self.out.grad.broadcast_to(x.shape)

class Reshape(Function):
  def forward(self, x: shrimp.Tensor, shape: Tuple[int,...] ) -> shrimp.Tensor:
    if prod(shape) != x.size: raise RuntimeError('shape \'{shape}\' is invalid for input of size {x.size}')
    self.out = shrimp.Tensor(shape, x.data, dtype=x.dtype)
    return self.out

  def backward(self):
    x = self.parents[0]
    return self.out.grad.reshape(*x.shape)

class Permute(Function):
  def forward(self, x: shrimp.Tensor, order: Tuple[int, ...]) -> shrimp.Tensor:
    self.order = order
    new_shape = [x.shape[i] for i in order]
    new_strides = [x.strides[i] for i in order]
    self.out = shrimp.Tensor(tuple(new_shape), x.data, dtype=x.dtype) 
    self.out.strides = new_strides
    return self.out
  
  def backward(self):
    return self.out.grad.permute(argsort(self.order))