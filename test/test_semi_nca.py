from typing import Callable, List
import unittest
from shrimpgrad.engine.semi_nca import init_idom, run_dfs, semi_nca, semidominators
from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.future import reverse_graph
from shrimpgrad.tensor import Tensor
from shrimpgrad.nn import Linear

class TestSemiNCA(unittest.TestCase):
  def test_rundfs(self):
    x = Tensor.randn(10,10)
    # LoadOps.EMPTY -> LoadOps.COPY
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y
    b = a * z
    c = b / w
    out = b - c

    g = reverse_graph(out.thunk)
    ln, ni, nn = run_dfs(g, out.thunk, 0, 0)
    self.assertEqual(4, ln)
    self.assertEqual(out.thunk, nn[0])
    self.assertEqual(b.thunk, nn[1])
    self.assertEqual(1, ni[out.thunk].DFSNum)
    self.assertEqual(1, ni[out.thunk].Label)
    self.assertEqual(1, ni[out.thunk].Semi)
    self.assertEqual(0, ni[out.thunk].Parent)

  def test_semi_nca_idominit(self):
    x = Tensor.randn(10,10)
    # LoadOps.EMPTY -> LoadOps.COPY
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y
    b = a * z
    c = b / w
    out = b - c
    g = reverse_graph(out.thunk)
    _, node_info, num_to_node = run_dfs(g, out.thunk, 0, 0)

    init_idom(len(num_to_node), node_info, num_to_node)
    for i in range(1, len(num_to_node)):
      self.assertEqual(num_to_node[node_info[num_to_node[i]].Parent],node_info[num_to_node[i]].IDom)

  def test_semidominators(self):
    x = Tensor.randn(10,10)
    # LoadOps.EMPTY -> LoadOps.COPY
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y
    b = a * z
    c = b / w
    out = b - c
    g = reverse_graph(out.thunk)
    _, node_info, num_to_node = run_dfs(g, out.thunk, 0, 0)
    num_to_info = init_idom(len(num_to_node), node_info, num_to_node)
    semidominators(len(num_to_node), num_to_info) 
    # TODO: Add asserts
  
  def test_semi_nca(self):
    x = Tensor.randn(10,10)
    # LoadOps.EMPTY -> LoadOps.COPY
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y
    b = a * z
    c = b / w
    out = b - c
    g = reverse_graph(out.thunk)
    _, node_info, num_to_node = run_dfs(g, out.thunk, -1, 0)
    semi_nca(g, out.thunk, node_info, num_to_node)
    # Because it is the root of the reverse graph it has no dominator
    self.assertEqual(node_info[out.thunk].IDom, out.thunk)
    self.assertEqual(node_info[c.thunk].IDom, out.thunk)
    self.assertEqual(node_info[b.thunk].IDom, out.thunk)
    self.assertEqual(node_info[a.thunk].IDom, b.thunk)
  
  def test_semi_nca_basic_mlp(self):
    class Model:
      def __init__(self):
        self.layers: List[Callable[[Tensor], Tensor]] = [
          Linear(2, 50), Tensor.relu,
          Linear(50, 1), Tensor.sigmoid, 
        ]
      def __call__(self, x: Tensor):
        return x.sequential(self.layers)
    model = Model()
    x = Tensor.randn(100,2)
    out = model(x)
    log_thunk(out.thunk)
    g = reverse_graph(out.thunk)
    _, node_info, num_to_node = run_dfs(g, out.thunk, -1, 0)
    semi_nca(g, out.thunk, node_info, num_to_node)
    self.assertEqual(node_info[out.thunk].IDom, out.thunk)

    for thunk, childs in g.items(): 
      # This is a linear graph (without the loads)
      # Exit node has no children
      if childs:
        self.assertEqual(1, len(childs))
        self.assertEqual(thunk, node_info[childs[0]].IDom)