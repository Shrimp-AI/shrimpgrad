from collections import deque
from typing import Deque, List, Optional
from shrimpgrad.engine.postdomtree import NodeToInfo, NumToNode, ipdoms
from shrimpgrad.future import Thunk 
from shrimpgrad.runtime.ops import AlgebraicOp 

class FusedOp:
  # Injective Operators are BinaryOps and UnaryOps and TernaryOps
  # Reduction are ReduceOps
  # Rules for Fusion:
  # 1. Injective operators can fuse with other injective operators
  # 2. Reduction operators can fuse with input injective operators to form a fused
  #    injective operator 

  # Fusion always starts with an injective
  # since we can't fuse multiple reductions
  def __init__(self, injective_thunk: Thunk):
    self.reduce: Optional[Thunk] = None
    self.injectives: List[Thunk] = [injective_thunk]
    self.shape = injective_thunk.shape

  @property
  def has_reduce(self): return self.reduce is not None

  def can_fuse(self, thunk: Thunk) -> bool:
    # Chained injectives preserve shape due to broadcasting
    # Reduction ops will have the input shape equal to the output shape of the previous injective op
    # if there was one. Axis will reduce the shape post execution hence we can fuse.
    if thunk.isreduce and thunk.reduce_input_shape == self.shape: return True
    if thunk.shape != self.shape: return False
    # Can always fuse injectives if the shapes match
    if thunk.algebraic_op is AlgebraicOp.INJECTIVE: return True
    # Can only fuse reductions if there is not a reduction in this FusedOp
    if thunk.algebraic_op is AlgebraicOp.REDUCTION and not self.has_reduce: return True
    # Can't fuse NOOPs (shouldn't be in the reduced reverse graph anyway but just in  case)
    return False

  # Returns True on fusion and False on non-fusion
  def fuse(self, thunk: Thunk) -> bool:
    if self.can_fuse(thunk):  
      if thunk.algebraic_op is AlgebraicOp.INJECTIVE: self.injectives.append(thunk)
      else: self.reduce = thunk
      return True
    return False

class FusionEngine:
  def __init__(self, thunk: Thunk): 
    self.root = thunk
    self.ipdoms, self.node_to_info, self.node_to_num = ipdoms(self.root)
    # Real roots are operation thunks that only have loads as inputs
    self.real_roots = [node for node, ninfo in self.node_to_info.items() if ninfo.Indegree == 0]
  
  def _init_groups(self):
    # Initial groups are self-contained subgraphs from real roots
    # to their ipdoms
    self.groups: Deque[List[Thunk]] = deque() 
    # TODO: Think about overlapping groups, whichever starts to fuse first will
    # remain fused. 
    for real_root in self.real_roots:
      # Can't start with a reduction
      if real_root.algebraic_op is AlgebraicOp.REDUCTION:
        continue
      group = self._gen_group(real_root) 
      self.groups.append(group)

  def _gen_group(self, root: Thunk) -> List[Thunk]:
      return bfs(self.node_to_info, self.node_to_num, root, self.ipdoms[root])
 
  def start(self) -> List[FusedOp]:
    # Start fusing
    #
    self._init_groups()
    # Start trying to fuse the first group 
    # If fusion succeeds, build a graph from end of the last group to the ipdoms and try to fuse it
    # If fusion fails early (the whole group is not consumed), commit the fusedop
    #   then build a group from end to its ipdoms
    #   then start fusing that group, continue until all groups are fused
    # Should we attempt to fuse all real roots first (BFS?) or DFS fuse?
    commited: List[FusedOp] = []
    print(f"STARTING {self.groups}")
    while self.groups:
      # Favor earlier groups (we could dynamically adjust this to find optimal fuses)
      group = self.groups.popleft()
      print(f"FUSING {group}")
      fused_op = None
      leader = None
      for i, thunk in enumerate(group):
        if thunk.algebraic_op is AlgebraicOp.INJECTIVE:
          fused_op = FusedOp(thunk)
          leader = i 
          break
        print(f"Cant fuse {thunk} not a valid leader")
      if leader is None:
        print(f"Cant fuse group {group} because no valid leader exists")
        continue
      if leader + 1 == len(group):
        print("Leader is ipdom, can't fuse group.")
        next_group = self._gen_group(group[leader])
        if next_group: self.groups.append(next_group)
        continue 
      print("Attempting to fuse")
      def fuse(group, leader, fused_op):
        nonlocal commited
        while True:
          has_fuse_chain = False
          if leader >=len(group):
            print("DONE")
            break
          last_fused = None 
          for candidate in group[leader+1:]:
            print(f"Fusing {candidate}")
            did_fuse = fused_op.fuse(candidate)
            if did_fuse:
              print(f"Fused sucessfully {candidate} with {fused_op}")
              # We have a chain of fuses
              has_fuse_chain = True
              last_fused = candidate
            else:
              print(f"Failed to fuse from {candidate} to {fused_op}")
              leader = group.index(candidate)
              if has_fuse_chain:
                print(f"Committing fused op {fused_op}")
                commited.append(fused_op)
                print(f"Update leader to candidate {leader}")
                has_fuse_chain = False
              else:
                print(f"No fusion chain, update leader to next node {leader}")
              last_fused = None
              break
          print("Group processed...")
          if has_fuse_chain:
              # if current fused op has a reduce we are done
              # else (create another group from the last fused op to it's ipdom)
              print("We have a fuse chain...") 
              next_group = None
              if last_fused is not None:
                next_group = self._gen_group(last_fused)
              if next_group is not None and next_group:
                print(f"Continuing to fuse with {next_group}")
                fuse(next_group, 0, fused_op)
              else:
                print(f"Committing fused op {fused_op}")
                commited.append(fused_op)
          print(f"End Fusion - Group {group} exhausted")
          break
      fuse(group, leader, fused_op)
    return commited

def bfs(node_info: NodeToInfo, num_to_node: NumToNode, start: Thunk, end: Thunk) -> List[Thunk]:
  if start == end: return []
  frontier = deque([start])
  visited = set()
  group = []
  while frontier:
    node = frontier.popleft()
    if node in visited: continue
    if node == end: 
      group.append(node)
      break
    visited.add(node)
    group.append(node)
    for childid in node_info[node].ReverseChildren: 
      child = num_to_node[childid]
      if child not in visited:
        frontier.append(child)
  return group
