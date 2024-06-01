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

    g = IndexedForwardGraph(e.thunk)
    print(g.ordering) 
    expected_ordering = [e.thunk.base, d.thunk.base, c.thunk.base, b.thunk.base, a.thunk.base]
    self.assertEqual(expected_ordering, list(g.ordering))
    for i, node in enumerate(expected_ordering):
      self.assertEqual(g.node2num(node), i)
    self.assertTrue(all([t in g.saved[0] for t in [x.thunk, y.thunk, z.thunk, w.thunk]]))
  
  def test_saved_expand(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    a = x + y
    b = x * a
    c = b.sum().expand(10,10)
    d = c / b
    e = d.mean()
    log_thunk(e.thunk)
    g = IndexedForwardGraph(e.thunk)

    print(g.saved)

    # self.assertTrue(all([t in g.saved[0] for t in [x.thunk, y.thunk]]))
    # self.assertTrue(len(g.saved[1]), 1)
    # self.assertTrue(g.saved[1][0].shape == (10,10))
    # self.assertEqual(g.saved[1][0].base.shape, (1,1))
    for i, node in enumerate(g.ordering):
      self.assertEqual(g.node2num(node), i) 