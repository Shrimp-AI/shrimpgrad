from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, TypeAlias
from shrimpgrad.future import Thunk, ThunkGraph, preorder_with_preds, reverse_graph

class UnionFind:
  def __init__(self, n):
    self.root = [i for i in range(n)]
    self.rank = [1] * n

  def union(self, u, v):
    root_u, root_v = self.find(u), self.find(v)
    if root_u != root_v:
      if self.rank[root_u] < self.rank[root_v]: self.root[root_u] = root_v
      elif self.rank[root_u] > self.rank[root_v]: self.root[root_v] = root_u
      else: 
        self.root[root_v] = root_u
        self.rank[root_u] += 1

  def find(self, u):
    if u == self.root[u]: return u
    self.root[u] = self.find(self.root[u])
    return self.root[u]

def semidominator(thunk: Thunk): 
  rg = reverse_graph(thunk)
  rpo = preorder_with_preds(thunk, rg, preds:=defaultdict(list), parent:={}, reverse=True)
  rev_idx = {p:i for i, p in enumerate(rpo)}
  uf = UnionFind(len(rpo))
  semi = [i for i in range(len(rpo))]
  for i, t in enumerate(rpo):
    if t == thunk: continue
    for p in preds[t]:
      z = uf.find(rev_idx[p])
      if semi[z] < semi[rev_idx[t]]: semi[rev_idx[t]] = semi[z]
    uf.union(rev_idx[t], rev_idx[parent[t]])
  return {rpo[u]:rpo[s] for u, s in enumerate(semi)}

@dataclass(frozen=False)
class InfoRec:
  DFSNum: int  = 0
  Parent: int = 0
  Semi: int = 0
  Label: int = 0
  ReverseChildren: List[int] =() 
  IDom = None

NodeToInfo: TypeAlias =  Dict[Thunk, InfoRec]
NumToNode: List[Thunk] = []
  
def runDFS(G: ThunkGraph, v: Thunk, last_num:int, attach_to_num: int):
  visited = set()
  node_info: NodeToInfo = defaultdict(InfoRec)
  assert(v)
  WorkList = [v]
  while len(WorkList):
    BB: Thunk = WorkList.pop()
    BBInfo: InfoRec = node_info[BB]
    if BBInfo.DFSNum != 0: continue 
    visited.add(BB)
    BBInfo.DFSNum = BBInfo.Semi = BBInfo.Label = last_num + 1
    last_num += 1
    NumToNode.append(BB)
    print(G[BB])
    for succ in G[BB]:
      print(succ)
      succ_info = node_info[succ]
      if succ in visited:
        succ_info.ReverseChildren.append(succ)
        continue

      WorkList.append(succ)
      succ_info.Parent = last_num
      childs = list(succ_info.ReverseChildren)
      childs.append(last_num)
      succ_info.ReverseChildren = tuple(childs)
  return last_num, node_info, NumToNode


