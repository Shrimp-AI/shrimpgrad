from typing import Iterable
from shrimpgrad.dtype import dtypes
from shrimpgrad.tensor import Tensor

class Optimizer:
  def __init__(self, params: Iterable[Tensor], lr=1e-3):
    assert params, 'params are empty'
    assert not (lr < 0.0), 'lr must be positive'
    for param in params: 
      if param.requires_grad is None: param.requires_grad = True 
    self.params = [x for x in params if x.requires_grad] 
    assert len(self.params) > 0, "optimizer requires at least one parameter"
    self.lr = lr

  def step(self): raise NotImplementedError('implement step')
  def zero_grad(self):
    for param in self.params: param.grad = None

class SGD(Optimizer):
  def __init__(self, params: Iterable[Tensor], lr=1e-3, momentum:float=0.0, dampening:float=0.0, weight_decay:float=0.0, nesterov:bool=False):
    super().__init__(params, lr)
    assert not (momentum < 0.0), 'momentum must be positive'
    assert not (weight_decay < 0.0), 'weight_decay must be positive'
    self.momentum, self.dampening, self.weight_decay, self.nesterov = momentum, dampening, weight_decay, nesterov
    self.b = [Tensor.zeros(t.shape, dtype=dtypes.float32, device=t.device, requires_grad=False).contiguous() for t in self.params]

  def step(self):
    for i,t in enumerate(self.params):
      g = t.grad
      assert g is not None, 'gradient cannot be empty for parameter in SGD'
      if self.weight_decay != 0.0:
        g += self.weight_decay*t
      if self.momentum != 0:
        # b[i] is zero initialized
        self.b[i].assign(self.b[i]*self.momentum + (1. - self.dampening)*g)
        if self.nesterov:
          g += self.momentum*self.b[i]
        else:
          g = self.b[i]
      g = self.lr * g
      t.assign(t.detach() - g)
      t.realize()

class Adam(Optimizer):
  def __init__(self, params: Iterable[Tensor], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay:float=0.0, amsgrad:bool=False):
    super().__init__(params, lr)
    self.betas, self.eps, self.weight_decay, self.amsgrad = betas, eps, weight_decay, amsgrad
    self.m = [Tensor.zeros(t.shape, dtype=dtypes.float32, device=t.device, requires_grad=False).contiguous() for t in self.params]
    self.v = [Tensor.zeros(t.shape, dtype=dtypes.float32, device=t.device, requires_grad=False).contiguous() for t in self.params]
    self.b1t = 1. 
    self.b2t = 1.

  def step(self):
    self.b1t *= self.betas[0]
    self.b2t *= self.betas[1]
    for i,p in enumerate(self.params):
      g = p.grad
      assert g is not None, 'gradient cannot be empty for parameter in Adam'
      if self.weight_decay != 0.0:
        g += self.weight_decay*p.detach()
      self.m[i].assign(self.betas[0]*self.m[i] + (1. - self.betas[0]) * g)
      self.v[i].assign(self.betas[1]*self.v[i] + (1. - self.betas[1]) * g.square())
      mhat = self.m[i] / (1.- self.b1t)
      vhat = self.v[i] / (1. - self.b2t)
      p.assign(p.detach() - self.lr * mhat / (vhat.sqrt() + self.eps))
      p.realize()