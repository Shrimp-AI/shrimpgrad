import unittest
from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.future import IndexedForwardGraph
from shrimpgrad.tensor import Tensor


class TestIndexedForwardGraph(unittest.TestCase):
  def test_diamond_graph(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y

    b = z * a 

    c = w * a 

    d = b / c

    e = d.sum()

    IndexedForwardGraph(e.thunk)
  
  def test_saved_expand(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum().expand(10,10)
    d = c / b
    e = d.mean()
    log_thunk(e.thunk)
    IndexedForwardGraph(e.thunk)