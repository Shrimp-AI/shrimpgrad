import math


class Value:

  def __init__(self, data, _children=(), _op='', label='', grad=0.0):
    # Public
    self.data = data
    self.grad = grad
    self.label = label
    self.prev = list(_children)
    # Private
    self._op = _op
    self._backward = lambda: None

  def backward(self):
    self.grad = 1.0
    visited = set()
    topo = []

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for c in v.prev:
          build_topo(c)
        topo.append(v)
    build_topo(self)
    for v in reversed(topo):
      v._backward()

  def __add__(self, other):
    other = other if isinstance(
      other, Value) else Value(other, label=f'{other}')
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    other = other if isinstance(
      other, Value) else Value(other, label=f'{other}')
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __pow__(self, other):
    assert(isinstance(other, (int, float))), "only support power to int/float"
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out

  def __truediv__(self, other):
    return self * (other ** - 1)

  def tanh(self):
    x = 2 * self
    return (x.exp() - 1) / (x.exp() + 1)

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), f'e^{x:0.4f}')

    def _backward():
      self.grad += math.exp(x) * out.grad
    out._backward = _backward
    return out

  def clear(self):
    self.grad = 0.0
    for c in self.prev:
      c.clear()

  def __repr__(self):
    return f'Value(data={self.data})'
