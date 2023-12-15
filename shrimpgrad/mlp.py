import numpy as np
from shrimpgrad.value import Value


class Neuron:
  def __init__(self, nin):
    self.w = [Value(wi) for wi in np.random.uniform(
      low=-1.0, high=-1.0, size=(nin,))]
    self.b = Value(np.random.uniform(low=-1.0, high=1.0))

  def __call__(self, x):
    return (sum([wi * xi for xi, wi in zip(x, self.w)]) + self.b).tanh()

  def parameters(self):
    return self.w + [self.b]


class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    return [n(x) for n in self.neurons]

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]


class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x[0] if len(x) == 1 else x

  def parameters(self):
    return [p for l in self.layers for p in l.parameters()]
