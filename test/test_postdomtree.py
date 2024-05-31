import unittest

from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.future import IndexedForwardGraph
from shrimpgrad.tensor import Tensor
from shrimpgrad.engine.postdomtree import PostDomTree 


class TestLCAPostDom(unittest.TestCase):

  def test_diamond_ipdom(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y

    b = z * a 

    c = w * a 

    d = b / c
    g = IndexedForwardGraph(d.thunk)
    log_thunk(d.thunk)
    ipdom_tree = PostDomTree(g) 
    # Show all ipdoms are DIV (d.thunk)
    i = ipdom_tree.node2num(d.thunk) 
    ipdom = ipdom_tree.tree[i]
    self.assertIs(None, ipdom.parent)

    i = ipdom_tree.node2num(c.thunk)
    ipdom = ipdom_tree.tree[i]
    self.assertEqual(ipdom.parent.thunk, d.thunk)

    i = ipdom_tree.node2num(b.thunk)
    ipdom = ipdom_tree.tree[i]
    self.assertEqual(ipdom.parent.thunk, d.thunk)

    
    i = ipdom_tree.node2num(a.thunk)
    ipdom = ipdom_tree.tree[i]
    self.assertEqual(ipdom.parent.thunk, d.thunk)