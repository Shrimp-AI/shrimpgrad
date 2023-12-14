import math


class Value:
    def __init__(self, data, _children=(), _op='', label='', grad=0.0):
        # Public
        self.data = data
        self.grad = grad
        self.label = label
        # Private
        self._prev = list(_children)
        self._op = _op
        self._backward = lambda: None

    def backward(self):
        self.grad = 1.0
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build_topo(c)
                topo.append(v)
        build_topo(self)
        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def clear(self):
        self.grad = 0.0
        for c in self._prev:
            c.clear()

    def __repr__(self):
        return f'Value(data={self.data})'
