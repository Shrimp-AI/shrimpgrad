from collections import defaultdict
from shrimpgrad.future import Thunk, preorder_with_preds, reverse_graph

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