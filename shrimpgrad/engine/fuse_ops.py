from __future__ import annotations
from collections import defaultdict
from typing import Callable, DefaultDict, List, TypeAlias 
from shrimpgrad.engine.postdomtree import ipdoms
from shrimpgrad.future import Thunk, ThunkGraph 
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

  def count(self) -> int: 
    count = 1 
    p = self
    while p.parent is not None:
      p = p.parent
      count += 1
    return count
      
  def __repr__(self):
    return f"<Group root={self.root} parent={self.parent}>"

class FusionEngine:
  def __init__(self, thunk: Thunk): 
    self.root = thunk
    self.ipdoms, self.node_to_info, self.num_to_node = ipdoms(self.root)
    print(self.num_to_node)
    print(self.node_to_info)
    self.groups: List[Group] = [Group(node.algebraic_op, node) for node in self.num_to_node]
    print(f"group length={len(self.groups)}")

  def fuse(self) -> ThunkGraph: 
    for i, group in enumerate(self.groups):
      print(f"Fusing group={group}")
      thunk = self.num_to_node[i]
      print(f"   src={thunk}")
      ipdom = self.ipdoms[thunk]
      print(f"   ipdom={ipdom}")
      ipdom_group = self.groups[self.node_to_info[ipdom].DFSNum]
      print(f"   ipdom_group={ipdom_group}")
      if ipdom_group.find_root() == group.find_root(): 
        print("   Group roots are equal, moving to next group.")
        continue     
      print("Groups are fusable")
      if thunk.algebraic_op == AlgebraicOp.INJECTIVE:
        # check that all intermediate thunks are injective or if it's the sink, a reduction or injective 
        print("Checking fuse condition...")
        fcnd: FuseCondition = lambda kind, is_sink: kind == AlgebraicOp.INJECTIVE or (is_sink and kind == AlgebraicOp.REDUCTION) 
        if(self.check_path(thunk, ipdom, fcnd)):
          print("Path is fuse valid...")
          self.commit_fuse(thunk, ipdom)
    return self.aggregate_groups()
  
  def aggregate_groups(self) -> ThunkGraph:
    gmap: ThunkGraph = defaultdict(list)
    for group in self.groups:
      if group.parent is None: continue
      gmap[group.parent.root].append(group.root)
    return gmap

  def check_path(self, src: Thunk, sink: Thunk, fcnd: FuseCondition):
    assert src != sink, "check path requires src and sink to be different"
    visited = set() 
    def _check_path(src: Thunk, sink: Thunk) -> bool:
      if src in visited: return True
      visited.add(src)
      group = self.groups[self.node_to_info[src].DFSNum]
      if not fcnd(group.aop, src == sink): return False
      if src == sink: return True
      for childid in self.node_to_info[src].ReverseChildren:
        child = self.num_to_node[childid]
        if not _check_path(child, sink): return False
      return True
    for childid in self.node_to_info[src].ReverseChildren:
      child = self.num_to_node[childid]
      if not _check_path(child, sink): return False
    return True

  def commit_fuse(self, src: Thunk, sink: Thunk):
    assert src != sink, "commit_fuse requires src and sink to be different"
    visited = set()
    target = self.groups[self.node_to_info[sink].DFSNum]
    def _commit_fuse(src: Thunk, sink: Thunk, target: Group):
      print(f"Commiting from {src} to {sink}")  
      if src == sink or src in visited: 
        print("End commit...")
        return
      visited.add(src)
      group = self.groups[self.node_to_info[src].DFSNum]
      print("Combining groups...")
      group.union(target)
      for childid in self.node_to_info[src].ReverseChildren:
        child = self.num_to_node[childid]
        _commit_fuse(child, sink, target)
    _commit_fuse(src, sink, target)