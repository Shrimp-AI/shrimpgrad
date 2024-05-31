from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set
from shrimpgrad.future import IndexedForwardGraph, Thunk, post_dfs, forward_graph 


@dataclass
class DomTreeNode:
  parent: DomTreeNode
  depth: int
  thunk: Thunk

class PostDomTree:
  def __init__(self, graph: IndexedForwardGraph):
    self.graph = graph
    self.tree: DomTreeNode = [None] * len(self.graph.ordering)
    self.node_to_num = self.graph.node_to_num
    self.dfs_post_order = self.graph.ordering
    for i, node in enumerate(self.graph.ordering):
      self.tree[i] = self.get_node(node)

  def ipdom(self, thunk: Thunk) -> Thunk:
    dtn: DomTreeNode = self.tree[self.node2num(thunk)]
    if dtn.parent is None: return thunk
    return dtn.parent.thunk
  
  def node2num(self, node: Thunk) -> int:
    return self.graph.node2num(node)

  def get_node(self, node: Thunk) -> DomTreeNode:
    # output node
    print(f"  tree={self.tree}")
    print(f"  node={node} ")
    if self.tree[0] is None:
      tnode = DomTreeNode(None, 1, node)
    else:
      children = self.graph.G[node]
      parent = self.lca(children)
      tnode = DomTreeNode(parent, parent.depth + 1 if parent else 1, node)
    return tnode

  def lca(self, input_nodes: List[Thunk]) -> Optional[DomTreeNode]:
    if not input_nodes: return None
    node = input_nodes[0]
    parent = self.tree[self.node2num(node)]
    for next_node in input_nodes[1:]:
      parent = self.lca_(parent,  self.tree[self.node2num(next_node)])
    return parent
  
  def lca_(self, lhs: DomTreeNode, rhs: DomTreeNode) -> DomTreeNode:
    while lhs != rhs:
      if lhs is None: return None
      if rhs is None: return None
      if lhs.depth < rhs.depth:
        rhs = rhs.parent
      elif rhs.depth < lhs.depth:
        lhs = lhs.parent
      else:
        # They are at the same depth
        # so lower both
        lhs, rhs = lhs.parent, rhs.parent
    return lhs
