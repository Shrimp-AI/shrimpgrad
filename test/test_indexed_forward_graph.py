import unittest
from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.future import Thunk, IndexedForwardGraph
from shrimpgrad.tensor import Tensor


class TestIndexedForwardGraph(unittest.TestCase):
  def test_post_order_dfs(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y

    b = z * a 

    c = w * a 

    d = b / c

    e = d.sum()
    log_thunk(e.thunk)

    g = IndexedForwardGraph(e.thunk)

    # print(g.ordering)
    # print(g.G)
    print(g.saved)