from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set
from shrimpgrad.future import Thunk, post_dfs, forward_graph 


@dataclass
class DomTreeNode:
  parent: DomTreeNode
  depth: int
  thunk: Thunk

class PostDomTree:
  def __init__(self, out: Thunk):
    self.out = out
    self.graph, self.roots = forward_graph(self.out)
    self.dfs_post_order, self.node_to_num = post_dfs(self.graph, self.roots[0], reverse=False) 
    self.tree: DomTreeNode = [None] * len(self.dfs_post_order)

    for i, node in enumerate(self.dfs_post_order):
      self.tree[i] = self.get_node(node)

  def ipdom(self, thunk: Thunk) -> Thunk:
    dtn: DomTreeNode = self.tree[self.node_to_num[thunk]]
    if dtn.parent is None: return thunk
    return dtn.parent.thunk

  def get_node(self, node: Thunk) -> DomTreeNode:
    # output node
    if self.tree[0] is None:
      tnode = DomTreeNode(None, 1, node)
    else:
      children = self.graph[node]
      print(f"CHILDS {children}")
      parent = self.lca(children)
      tnode = DomTreeNode(parent, parent.depth + 1 if parent else 1, node)
    return tnode

  def lca(self, input_nodes: List[Thunk]) -> Optional[DomTreeNode]:
    if not input_nodes: return None
    node = input_nodes[0]
    parent = self.tree[self.node_to_num[node]]
    for next_node in input_nodes[1:]:
      parent = self.lca_(parent,  self.tree[self.node_to_num[next_node]])
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
