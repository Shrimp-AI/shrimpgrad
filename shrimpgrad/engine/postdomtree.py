from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeAlias
from shrimpgrad.future import Thunk, ThunkGraph, reverse_graph

# With help from https://llvm.org/doxygen/GenericDomTreeConstruction_8h_source.html
@dataclass
class InfoRec:
  DFSNum: int  = 0
  Parent: int = 0
  Semi: int = 0
  Label: int = 0
  Indegree: int = 0
  ReverseChildren: List[int] =()
  IDom: Optional[Thunk] = None

NodeToInfo: TypeAlias =  Dict[Thunk, InfoRec]
NumToNode: TypeAlias = List[Thunk] 
NumToInfo: TypeAlias = List[InfoRec]

# Iterative DFS (defaults to reverse pre-order traversal (right then left))  
# Requires calling reverse_graph on the output thunk to generate G
# returns the last dfs number, node info and the reverse index from pre-order number to node
def run_dfs(G: ThunkGraph, v: Thunk, last_num:int, attach_to_num: int, reverse=True):
  node_info: NodeToInfo = defaultdict(InfoRec)
  num_to_node: NumToNode = []
  assert(v)
  WorkList = [v]
  while len(WorkList):
    BB: Thunk = WorkList.pop()
    BBInfo: InfoRec = node_info[BB]
    if BBInfo.DFSNum != 0: continue 
    BBInfo.DFSNum = BBInfo.Semi = BBInfo.Label = last_num + 1
    last_num += 1
    num_to_node.append(BB)
    successors = G[BB] if not reverse else G[BB][::-1]
    for succ in successors:
      succ_info = node_info[succ]
      if succ_info.DFSNum != 0:
        if succ != BB:
          childs = list(succ_info.ReverseChildren)
          childs.append(last_num)
          succ_info.ReverseChildren = tuple(childs)
          BBInfo.Indegree += 1
        continue
      WorkList.append(succ)
      succ_info.Parent = last_num
      childs = list(succ_info.ReverseChildren)
      childs.append(last_num)
      succ_info.ReverseChildren = tuple(childs)
      # Since we are pointing towards the original graphs root
      # In actuality the indegree points toward the parent from the child
      BBInfo.Indegree += 1
  return last_num, node_info, num_to_node  

# Initialize Immediate Dominators to be the parent of each node
def init_idom(next_dfs_num: int, node_info: NodeToInfo, num_to_node: List[Thunk]) -> NumToInfo:
  num_to_info: List[InfoRec] = []
  for i in range(0, next_dfs_num):
    v = num_to_node[i]
    v_info = node_info[v]
    v_info.IDom = num_to_node[v_info.Parent]
    num_to_info.append(v_info)
  return num_to_info

def seval(v: int, last_linked: int, stack: List[InfoRec], num_to_info:NumToInfo):
  v_info = num_to_info[v]
  if v_info.Parent < last_linked: return v_info.Label
  assert(not len(stack))

  while True:
    stack.append(v_info)
    v_info = num_to_info[v_info.Parent]
    if not v_info.Parent >= last_linked:
      break
  p_info = v_info
  p_label_info = num_to_info[p_info.Label]
  while True:
    v_info = stack.pop()
    v_info.Parent = p_info.Parent 
    v_label_info = num_to_info[v_info.Label]
    if p_label_info.Semi < v_label_info.Semi:
      v_info.Semi = p_info.Label
    else:
      p_label_info = v_label_info
    p_info = v_info
    if not len(stack):
      break
  return v_info.Label

def semidominators(next_dfs_num: int, num_to_info: NumToInfo):
  eval_stack = []
  for i in range(next_dfs_num-2, 1, -1):
    w_info = num_to_info[i]
    # initialize the semidominator to point to w.Parent
    # since semidominators are proper ancestors
    w_info.Semi = w_info.Parent
    for n in w_info.ReverseChildren:
      semi_u = num_to_info[seval(n, i + 1, eval_stack, num_to_info)].Semi
      if semi_u < w_info.Semi: w_info.Semi = semi_u

def semi_nca(G: ThunkGraph, root: Thunk, node_info: NodeToInfo, num_to_node: List[Thunk]):
  next_dfs_num = len(num_to_node)
  num_to_info = init_idom(next_dfs_num, node_info, num_to_node)

  # Step #1: Calculate the semidominators of all vertices.
  semidominators(next_dfs_num, num_to_info)

  # Step #2: Explicitly define the immediate dominator of each vertex.
  #          IDom[i] = NCA(SDom[i], SpanningTreeParent(i)).
  # Note that the parents were stored in IDoms and later got invalidated
  # during path compression in Eval.
  for i in range(1, next_dfs_num):
    winfo = num_to_info[i]
    assert(winfo.Semi != 0)
    sdom_num = num_to_info[winfo.Semi].DFSNum
    widomcandidate = winfo.IDom
    while True:
      widomcandidateinfo = node_info[widomcandidate]
      if widomcandidateinfo.DFSNum <= sdom_num: break
      widomcandidate = widomcandidateinfo.IDom
    winfo.IDom = widomcandidate

IDoms: TypeAlias = Dict[Thunk, Thunk]
def ipdoms(out: Thunk) -> Tuple[IDoms, NodeToInfo, NumToNode]: 
  if out.isview: out = out.base
  g = reverse_graph(out)
  _, node_info, num_to_node = run_dfs(g, out, -1, 0)
  semi_nca(g, out, node_info, num_to_node)
  return {thunk:info.IDom for thunk, info in node_info.items()}, node_info, num_to_node
