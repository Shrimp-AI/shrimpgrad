"""
Module providing a scalar Value object 
that supports operations with reverse automatic differentiation
"""
import math


class Value:
    """Class representing a scalar value with autograd"""

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
        """Method that executes backpropagation starting at self."""
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
            v.backward_intern()

    def backward_intern(self):
        """Get the internal backward function
        implementing the backward pass for the op.

        Returns:
            Function: the backwards function
        """
        return self._backward

    def set_backward_intern(self, bw):
        """Set the internal backward pass for the op.

        Args:
            bw (Function): the backwards function
        """
        self._backward = bw

    def get_op(self):
        """Return the op used to evaluate this Value object

        Returns:
            string: the op as a string (i.e. '+', '*', 'tanh', etc.)
        """
        return self._op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        """Computes hyperbolic tangent (tanh) for the value.

        Returns:
            Value: the value after computing tanh
        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out. set_backward_intern(_backward)
        return out

    def exp(self):
        """Computes the exponential function for the value.

        Returns:
            Value: the value after e^x
        """
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad = math.exp(x) * out.grad
        out. set_backward_intern(_backward)
        return out

    def clear(self):
        """Clear all the gradients in the graph.
        """
        self.grad = 0.0
        for c in self.prev:
            c.clear()

    def __repr__(self):
        return f'Value(data={self.data})'
