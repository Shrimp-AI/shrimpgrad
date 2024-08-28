from typing import List
import math
from shrimpgrad import Tensor
from shrimpgrad.nn.optim import *
from shrimpgrad.util import prod
from . import datasets

class Linear:
  def __init__(self, in_features: int, out_features: int, bias:bool=True):
    self.w = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    bound = 1. / math.sqrt(in_features)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound)

  def __call__(self, x:Tensor) -> Tensor:
    return x.linear(self.w.transpose(), self.bias)

  def parameters(self) -> List[Tensor]:
    return [self.w, self.bias]

def get_parameters(model) -> List[Tensor]:
  params = []
  for layer in model.layers:
    if getattr(layer, 'parameters', None):
      params+=layer.parameters()
  return params


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
      # NOTE: this can be precomputed for static inference. we expand it here so it fuses
      batch_invstd = self.running_var.reshape(*shape_mask).expand(*x.shape).add(self.eps).rsqrt()
    return x.batch_norm(self.weight, self.bias, batch_mean, batch_invstd)
BatchNorm2d = BatchNorm3d = BatchNorm

  
