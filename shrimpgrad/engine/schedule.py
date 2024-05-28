from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Set, Tuple
from shrimpgrad.device import Buffer
from shrimpgrad.dtype import DType
from shrimpgrad.future import Thunk
from shrimpgrad.runtime.ops import BufferOps, LoadOps, Op
from shrimpgrad.util import prod
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

def build_mlir(root: Thunk) -> MLIR:
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

@dataclass(frozen=True)
class ScheduledKernel:
  ast: MLIR
  inputs: List[Buffer]
  outputs: List[Buffer]

@dataclass(frozen=True)
class ThunkKernel:
  ast: MLIR
  inputs: Tuple[Thunk, ...]
  outputs: Tuple[Thunk, ...] 


class Scheduler:  
  def __init__(self, outs: List[Thunk]):
    self.outs = outs
    self.visited: Set[Thunk] = set()
    self.targets: Dict[Thunk, None] = {}

  def schedule(self) -> List[ScheduledKernel]:
    for out in self.outs: self.search_for_targets(out)

    prescheduled = {}
    for target in self.targets:
      if target.realized is not None or target._op is LoadOps.CONST: continue
      prescheduled[target] = self._preschedule(target)

    schedule_targets = {out:ps for ps in prescheduled.values() for out in ps.outputs}

    graph: DefaultDict[Thunk, List[Thunk]] = defaultdict(list)
    in_degree: DefaultDict[Thunk, int] = defaultdict(int)
    for target, sched_kernel in prescheduled.items(): 
      if target not in in_degree: in_degree[target] = 0

      # If any of the targets operands (parents) are also targets
      # we need to schedule them first in the DAG
      scheduled_parents = set(schedule_targets[parent].outputs[0] for parent in sched_kernel.inputs if parent in schedule_targets)

      # Update the graph such that parent kernels point to children kernels
      for parent in scheduled_parents:
        graph[parent].append(target)
        in_degree[target] += 1

    from collections import deque
    frontier = deque(pk for target, pk in prescheduled.items() if in_degree[target] == 0)

    schedule = []
    while frontier:
      pk = frontier.popleft()
      schedule.append(ScheduledKernel(ast=pk.ast, inputs=tuple(x.buff for x in pk.inputs if hasattr(x, 'buff')), outputs=tuple(x.buff for x in pk.outputs if hasattr(x, 'buff'))))
      for x in graph[pk.outputs[0]]:
        in_degree[x] -= 1
        if in_degree[x] == 0:
          frontier.append(prescheduled[x])
    return schedule 

  def search_for_targets(self, out: Thunk):
    # Targets are Loads and base of Views and Outputs
    # unrealized outputs nare targets 
    if out.base.realized is None: self.targets[out.base] = None
    # recurse on the output to find other targets (loads, view bases)
    self._search(out.base)

  def _search(self, thunk: Thunk):
    if thunk in self.visited or thunk.base.realized is not None: return
    # view: realize my base
    if thunk != thunk.base: 
      if prod(thunk.base.shape) < prod(thunk.shape):
        self.targets[thunk.base] = None
      return self._search(thunk.base)
    # base
    self.visited.add(thunk)
    if thunk._op in LoadOps: self.targets[thunk] = None
    if thunk._op is LoadOps.COPY:
      # Realize my operands (usually empty load) base
      self.targets[thunk._operands[0].base] = None
    for operand in thunk._operands:
      # recurse on my operands to find their targets 
      self._search(operand)

  def _preschedule(self, out):
    inputs: List[Thunk] = [] 
    if (out._op in {LoadOps.COPY, LoadOps.EMPTY, LoadOps.CUSTOM}): 
      inputs = [x.base for x in out._operands]
      return ThunkKernel(MLIR(out._op, (), out.arg), tuple(inputs), tuple([out]))
    return ThunkKernel(build_mlir(out), tuple(out._operands), tuple([out])) 

def _tree(mlir: MLIR, prefix="") -> str:
  if not len(mlir.inputs): return [f"━━ {prefix}{mlir.op.name} {mlir.arg if mlir.arg is not None else ''}"]
  lines = [f"━┳ {prefix}{mlir.op.name} {mlir.arg if mlir.arg is not None else ''}"]
  childs = [_tree(c) for c in mlir.inputs[:]]
  for c in childs[:-1]: lines += [f" ┣{c[0]}"] + [f" ┃{l}" for l in c[1:]]
  return lines + [" ┗"+childs[-1][0]] + ["  "+l for l in childs[-1][1:]]

def print_ast(mlir: MLIR): print("\n".join([f"{str(i).rjust(3)} {s}" for i,s in enumerate(_tree(mlir))]))
