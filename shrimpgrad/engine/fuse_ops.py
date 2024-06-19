from __future__ import annotations
from collections import defaultdict
from typing import Callable, List, TypeAlias
from shrimpgrad.engine.postdomtree import PostDomTree
from shrimpgrad.future import IndexedForwardGraph, Thunk, ThunkGraph
from shrimpgrad.runtime.ops import AlgebraicOp

FuseCondition: TypeAlias = Callable[[AlgebraicOp, bool], bool]

class Group:
  # A partition of the graph (union-find)
  def __init__(self, algebraic_op: AlgebraicOp, root: Thunk):
    self.aop, self.root = algebraic_op, root
    self.parent = None

  def find_root(self) -> Group:
    # find with path compression
    if self.parent is None: return self

    # Climb the parent chain
    root = self
    while root.parent is not None: root = root.parent
    # Update the parents to point to the new root
    p = self
    while p != root:
      parent = p.parent
      p.parent = root
      p = parent
    return root

  def union(self, parent) -> None:
    child = self.find_root()
    parent = parent.find_root()
    if child == parent: return
    child.parent = parent

  def __repr__(self): return f"<Group id={id(self)} root={self.root} parent={self.parent}>"

class FusionEngine:
  def __init__(self, g: IndexedForwardGraph):
    self.dom_tree = PostDomTree(g)

    self.num_to_node = self.dom_tree.dfs_post_order
    self.groups: List[Group] = [Group(node.algebraic_op, node) for node in self.num_to_node]

  def get_node_index(self, node: Thunk) -> int: return self.dom_tree.node2num(node)
  def get_children(self, node: Thunk) -> List[Thunk]: return self.dom_tree.graph.G[node]

  def fuse(self) -> ThunkGraph:
    for i, group in enumerate(self.groups):
      thunk = self.num_to_node[i]
      ipdom = self.dom_tree.ipdom(thunk)
      ipdom_group = self.groups[self.get_node_index(ipdom)]
      ipdom_root, group_root = ipdom_group.find_root(), group.find_root()
      if ipdom_root.root == group_root.root or ipdom_root == group_root: continue
      if thunk.algebraic_op == AlgebraicOp.INJECTIVE:
        # check that all intermediate thunks are injective or if it's the sink, a reduction or injective
        fcnd: FuseCondition = lambda kind, is_sink: kind == AlgebraicOp.INJECTIVE or (is_sink and kind == AlgebraicOp.REDUCTION)
        if(self.check_path(thunk, ipdom, fcnd)):
          self.commit_fuse(thunk, ipdom)
    return self.aggregate_groups()

  def aggregate_groups(self) -> ThunkGraph:
    gmap: ThunkGraph = defaultdict(list)
    for group in self.groups:
      if group.parent is None:
        continue
      gmap[group.parent.root].append(group.root)
    return gmap

  def check_path(self, src: Thunk, sink: Thunk, fcnd: FuseCondition):
    assert src != sink, "check path requires src and sink to be different"
    visited = set()
    def _check_path(src: Thunk, sink: Thunk) -> bool:
      if src in visited: return True
      visited.add(src)
      group = self.groups[self.get_node_index(src)]
      if not fcnd(group.aop, src == sink): return False
      if src == sink: return True
      for child in self.get_children(src):
        if not _check_path(child, sink): return False
      return True
    for child in self.get_children(src):
      if not _check_path(child, sink): return False
    return True

  def commit_fuse(self, src: Thunk, sink: Thunk):
    assert src != sink, "commit_fuse requires src and sink to be different"
    visited = set()
    target = self.groups[self.get_node_index(sink)]
    def _commit_fuse(src: Thunk, sink: Thunk, target: Group):
      if src == sink or src in visited:
        return
      visited.add(src)
      group = self.groups[self.get_node_index(src)]
      group.union(target)
      for child in self.get_children(src):
        _commit_fuse(child, sink, target)
    _commit_fuse(src, sink, target)