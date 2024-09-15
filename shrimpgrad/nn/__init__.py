from typing import List, Tuple
import math
from shrimpgrad import Tensor
from shrimpgrad.nn.optim import *
from shrimpgrad.util import prod

def get_parameters(model) -> List[Tensor]:
  params = []
  for layer in model.layers:
    if getattr(layer, 'parameters', None):
      params+=layer.parameters()
  return params

class Linear:
  def __init__(self, in_features: int, out_features: int, bias:bool=True):
    self.w = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    bound = 1. / math.sqrt(in_features)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound)

  def __call__(self, x:Tensor) -> Tensor:
    return x.linear(self.w.transpose(), self.bias)

  def parameters(self) -> List[Tensor]:
    return [self.w, self.bias]
  
class LayerNorm:
  def __init__(self, norm_shape: Tuple[int, ...]|int, eps: float = 1e-05, elementwise_affine: bool = True):
    self.eps = eps
    self.norm_shape = norm_shape if isinstance(norm_shape, tuple) else (norm_shape,)
    self.elementwise_affine = elementwise_affine
    self.weight = Tensor.ones(self.norm_shape) if elementwise_affine else None
    self.bias = Tensor.zeros(self.norm_shape) if elementwise_affine else None

  def __call__(self, x: Tensor) -> Tensor:
    assert x.shape[-len(self.norm_shape):] == self.norm_shape, f"expected last {len(self.norm_shape)} dims to be {self.norm_shape} but got {x.shape[-len(self.norm_shape):]}"
    axis = tuple([-i for i in range(len(self.norm_shape), 0, -1)])
    x = x - x.mean(axis=axis, keepdim=True)
    x = x / x.std(axis=axis, keepdim=True, correction=0)
    if self.elementwise_affine:
      assert self.weight is not None and self.bias is not None, "affine requires weight and bias"
      x = x * self.weight
      x = x + self.bias
    return x

class BatchNorm:
  def __init__(self, num_features:int, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
    sz = (num_features,)
    if affine: self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
    else: self.weight, self.bias = None, None
    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros((1,), requires_grad=False)

  def __call__(self, x:Tensor, training=False):
    shape_mask = [1, -1, *([1]*(x.ndim-2))]
    if training:
      batch_mean = x.mean(axis=(reduce_axes:=tuple(x for x in range(x.ndim) if x != 1)))
      y = (x - batch_mean.detach().reshape(*shape_mask))  # d(var)/d(mean) = 0
      batch_var = (y*y).mean(axis=reduce_axes)
      batch_invstd = batch_var.add(self.eps).rsqrt()

      # NOTE: wow, this is done all throughout training in most PyTorch models
      if self.track_running_stats:
        self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
        self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * prod(y.shape)/(prod(y.shape)-y.shape[1]) * batch_var.detach())
        self.num_batches_tracked += 1
    else:
      batch_mean = self.running_mean
      batch_invstd = self.running_var.reshape(*shape_mask).expand(*x.shape).add(self.eps).rsqrt()
    return x.batch_norm(self.weight, self.bias, batch_mean, batch_invstd)
  def parameters(self) -> List[Tensor]:
    return [self.weight, self.bias] if self.weight is not None and self.bias is not None else []

BatchNorm2d = BatchNorm3d = BatchNorm

class Conv2D:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|Tuple[int,int], bias:bool=True, stride:int|Tuple[int,int]=1, padding:int|Tuple[Tuple[int,int],...]=0, dilation:int|Tuple[int,int]=1, groups:int=1):
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
    scale = 1. / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale, requires_grad=True)
    self.bias = Tensor.uniform(out_channels, low=-scale, high=scale, requires_grad=True) if bias else None

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv2d(self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

  def parameters(self) -> List[Tensor]:
    return [self.weight, self.bias] if self.bias is not None else [self.weight]
