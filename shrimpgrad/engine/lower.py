from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
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
  return MLIR(op=BufferOps.STORE, inputs=(topo(root),), arg=buffs_mgr.get_mlir_buff(root))

def _tree(mlir: MLIR, prefix="") -> str:
  if not len(mlir.inputs): return [f"━━ {prefix}{mlir.op.name}"]
  lines = [f"━┳ {prefix}{mlir.op.name}"]
  childs = [_tree(c) for c in mlir.inputs[:]]
  for c in childs[:-1]: lines += [f" ┣{c[0]}"] + [f" ┃{l}" for l in c[1:]]
  return lines + [" ┗"+childs[-1][0]] + ["  "+l for l in childs[-1][1:]]

def print_ast(mlir: MLIR): print("\n".join([f"{str(i).rjust(3)} {s}" for i,s in enumerate(_tree(mlir))]))