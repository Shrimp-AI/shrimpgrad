from typing import List
import math
from shrimpgrad import Tensor
from shrimpgrad.nn.optim import *

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
