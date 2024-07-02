
from typing import Iterable
from shrimpgrad.tensor import Tensor

class Optimizer:
  def __init__(self, params: Iterable[Tensor], lr=1e-3):
    assert params, 'params are empty'
    assert not (lr < 0.0), 'lr must be positive'
    self.params = params
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

  def step(self):
    b = None
    for i,t in enumerate(self.params):
      g = t.grad
      # if g.thunk.base.realized is not None:
      #   print(f"Gradient: 0x{ctypes.addressof(g.thunk.base.buff._pointer(ctypes.c_float)):X}")
      assert g is not None, 'gradient cannot be empty for parameter in SGD'
      if self.weight_decay != 0.0:
        g += self.weight_decay*t
      if self.momentum != 0:
        if i > 0:
          b = b*self.momentum + (1 - self.dampening)*g
        else:
          b = g
        if self.nesterov:
          g += self.momentum*b
        else:
          g = b
      # if t.thunk.base.realized is not None:
      #   print(f"tensor before: {t.data()}")
      g = self.lr * g
      t.assign(t.detach() - g)
      t.realize()
      # if g.thunk.base.realized is not None:
      #   print(f"Gradient: 0x{ctypes.addressof(g.thunk.base.buff._pointer(ctypes.c_float)):X}")
      #   print(g.data())
      # print(f"tensor after: \n{t.data()}")
      if b is not None: b.realize()

