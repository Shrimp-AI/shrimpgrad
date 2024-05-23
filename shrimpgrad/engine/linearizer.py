from shrimpgrad.future import Thunk

def linearize(root: Thunk):
  results = []
  visited = set()
  def topo(root):
    if not root in visited:
      visited.add(root)
      if not len(root._operands) and root.base == root:
        results.append(root)
        return
      parents = root._operands if len(root._operands) else [root.base] 
      for p in parents: topo(p)
      results.append(root)
  topo(root)
  return results