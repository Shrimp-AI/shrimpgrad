from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Set, Tuple
from shrimpgrad.dtype import DType
from shrimpgrad.future import Thunk
from shrimpgrad.runtime.ops import BufferOps, LoadOps, Op
from shrimpgrad.view import View

@dataclass(frozen=True, eq=False)
class MLIR:
  # mid-level intermediate representation
  op: Op
  inputs: Tuple[MLIR, ...] = ()
  arg: Any = None

@dataclass(frozen=True)
class MLIRBuffer:
  index: int
  view: View
  dtype: DType

class BuffsManager:
  def __init__(self):
    self.buffs = {}
    self.idx = 0
  def get_mlir_buff(self, thunk: Thunk):
    if thunk.base.buff in self.buffs:
      return self.buffs[thunk.base.buff]
    buff = MLIRBuffer(index=self.idx, view=thunk._view, dtype=thunk.dtype)
    self.buffs[thunk.base.buff] = buff
    self.idx+=1
    return buff
  def real_buffs(self): 
    buffs = [None]*len(self.buffs)
    for rb, mb in self.buffs.items():
      buffs[mb.index] = rb 
    return buffs 

def lower(root: Thunk) -> MLIR:
  visited = set()
  buffs_mgr = BuffsManager()
  def topo(root, i=1):
    if not root in visited:
      visited.add(root)
      if root.base != root:
        v = root._view
        root = root.base
        root._view = v
      if root._op in LoadOps:
        arg = root.arg if root._op == LoadOps.CONST else buffs_mgr.get_mlir_buff(root)
        return MLIR(op=root._op, arg=arg)
      parents = root._operands 
      srcs = []
      for p in parents: 
        srcs.append(topo(p))
      return MLIR(op=root._op, inputs=srcs, arg=root.arg) 
  return MLIR(op=BufferOps.STORE, inputs=(topo(root),), arg=buffs_mgr.get_mlir_buff(root)), buffs_mgr.real_buffs()

def _recurse_thunk(thunk: Thunk, realizes: Dict[Thunk, None],allthunks: Dict[Thunk, None], children=Dict[Thunk, Dict[Thunk, None]], scheduled=False):
  if thunk in allthunks or thunk.base.realized is not None: return 
  if thunk.base != thunk:
    return _recurse_thunk(thunk.base, realizes, allthunks, children)
  # base
  allthunks[thunk] = None
  if thunk._op in LoadOps: realizes[thunk.base] = None
  for child in thunk._operands:
    children[child.base][thunk] = None
    _recurse_thunk(child, realizes, allthunks, children)

def graph_schedule(outs: List[Thunk], visited: Set[Thunk]):
  # realize output thunks
  realizes: Dict[Thunk, None] = {thunk.base:None for thunk in outs if thunk.realized is None}
  allthunks: Dict[Thunk, None] = {}
  children: DefaultDict[Thunk, Dict[Thunk, None]] = defaultdict(dict)
  for out in outs: _recurse_thunk(out, realizes, allthunks, children, scheduled=True)
  print(realizes)
  print(allthunks)
  print(children)

def _tree(mlir: MLIR, prefix="") -> str:
  if not len(mlir.inputs): return [f"━━ {prefix}{mlir.op.name} {mlir.arg if mlir.arg is not None else ''}"]
  lines = [f"━┳ {prefix}{mlir.op.name} {mlir.arg if mlir.arg is not None else ''}"]
  childs = [_tree(c) for c in mlir.inputs[:]]
  for c in childs[:-1]: lines += [f" ┣{c[0]}"] + [f" ┃{l}" for l in c[1:]]
  return lines + [" ┗"+childs[-1][0]] + ["  "+l for l in childs[-1][1:]]

def print_ast(mlir: MLIR): print("\n".join([f"{str(i).rjust(3)} {s}" for i,s in enumerate(_tree(mlir))]))