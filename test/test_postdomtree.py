import unittest

from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.future import forward_graph, post_dfs
from shrimpgrad.tensor import Tensor
from shrimpgrad.engine.postdomtree import PostDomTree 


class TestLCAPostDom(unittest.TestCase):
  def test_post_order(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y

    b = z * a 

    c = w * a 

    d = b / c
    log_thunk(d.thunk)
    g, roots = forward_graph(d.thunk)
    self.assertEqual(1, len(roots))
    expected_order = [d.thunk, b.thunk, c.thunk, a.thunk]
    post_dfs_order, node_to_num = post_dfs(g, roots[0], reverse=False)
    print(g)
    print(post_dfs_order)
    self.assertEqual(expected_order, post_dfs_order)
    for i, node in enumerate(post_dfs_order):
      self.assertEqual(i, node_to_num[node])


  def test_diamond_ipdom(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y

    b = z * a 

    c = w * a 

    d = b / c
    from pprint import pprint
    ipdom_tree = PostDomTree(d.thunk) 
    for node in ipdom_tree.tree:
      pprint(node)
