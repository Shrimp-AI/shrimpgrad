from __future__ import annotations
import base64
import math
import operator
import pickle
from typing import DefaultDict, List
from shrimpgrad.device import Accelerator, Allocator, Compiler, ConstBuffer, MemBuffer, Runtime
from shrimpgrad.engine.lower import ALUNode, ConstNode, GlobalNode, LocalNode, LowIR, LowIRGraph, Node
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps
from collections import deque
from functools import lru_cache

python_alu = {
  UnaryOps.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan,
  UnaryOps.EXP2: lambda x: math.exp(x*math.log(2)),
  UnaryOps.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, UnaryOps.SIN: math.sin,
  UnaryOps.NEG: lambda x: (not x) if isinstance(x, bool) else -x,
  BinaryOps.MUL: operator.mul, BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.XOR: operator.xor,
  BinaryOps.MAX: max, BinaryOps.CMPEQ: operator.eq, BinaryOps.CMPLT: operator.lt,
  BinaryOps.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0],
  BinaryOps.DIV: lambda x,y: int(x/y) if isinstance(x, int) else (x/y if y != 0 else x*math.inf),
  TernaryOps.WHERE: lambda x,y,z: y if x else z}

class PythonDevice(Accelerator):
  def __init__(self):
    super().__init__("PYTHON", PythonAllocator, PythonRenderer, PythonCompiler, PythonRuntime)
  def compiler(self) -> PythonCompiler: return self._compiler()
  def allocator(self) -> PythonAllocator: return self._allocator()
  def runtime(self) -> PythonRuntime: return self._runtime()
  def renderer(self) -> PythonRenderer: return self._renderer()
  def __repr__(self) -> str:
    return "<PythonDevice>"

class PythonRenderer:
  def render(self, ir_graph: LowIRGraph) -> str: return base64.b64encode(pickle.dumps(ir_graph)).decode()

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonRuntime(Runtime):
  def __init__(self):
    self.pc = 0
    self.global_scope = {}
    self.address = None
    self.load_stack = deque()

  def _loop(self, s: LocalNode ,e: int, instrs):
    loop_start = self.pc
    s_val = self.global_scope[s]
    self.pc+=1
    for _ in range(s_val,e):
      # Go to instruction after the for stmt
      end_loop_pc = self._exec(instrs)
      # Increment the backing value of the idx var
      self.global_scope[s] += 1
      self.pc = loop_start + 1
    return end_loop_pc

  def exec(self, lib: bytes, buffs: DefaultDict[str, List[MemBuffer | ConstBuffer]], buff2name):
    ir_graph: LowIRGraph = pickle.loads(lib)
    self.buffs = buffs
    self.buff2name = buff2name
    from cProfile import Profile
    from pstats import SortKey, Stats
    with Profile() as profile:
      self._exec(ir_graph.G)
      Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()

  def _exec(self, instrs: List[Node]):
    while self.pc < len(instrs):
      instr = instrs[self.pc]
      if instr.op is LowIR.END_LOOP:
        return self.pc

      if instr.op is LowIR.CONST:
        self.pc += 1
        continue
      elif instr.op is LowIR.LOCAL:
        if instr.ancestors[0].op is LowIR.ALU:
          alu_node = instr.ancestors[0]
          self.global_scope[instr] = self.exec_alu(alu_node)
        else:
          self.global_scope[instr] = instr.ancestors[0].val
      elif instr.op is LowIR.GLOBAL:
        buff = self.find_buff(instr.name)
        if isinstance(buff, MemBuffer):
          self.global_scope[instr] = buff.buff
        else:
          self.global_scope[instr] = buff.value
      elif instr.op is LowIR.BEGIN_LOOP:
        s, e = instr.ancestors[0], instr.ancestors[1].val
        end_loop_pc = self._loop(s, e, instrs)
        self.pc = end_loop_pc
      elif instr.op is LowIR.ADDRESS:
        idxs = instr.idx
        strides = instr.stride
        addr = 0
        for idx, stride in zip(idxs, strides):
          i = idx
          if idx.op is LowIR.LOCAL:
            i = self.global_scope[idx]
          addr += i * stride
        self.global_scope[instr] = addr
      elif instr.op is LowIR.LOAD:
        node = instr.ancestors[0]
        if node.op is LowIR.GLOBAL:
          buff = self.find_buff(node.name)
          if node.ptr:
            addr = self.global_scope[instr.ancestors[1]]
            loaded = buff.buff.pointer()[addr*4:addr*4+4].cast('f')[0]
          else:
            loaded = buff.value
          self.global_scope[instr] = loaded
      elif instr.op is LowIR.ALU:
        res = self.exec_alu(instr)
        self.global_scope[instr] = res
      elif instr.op is LowIR.STORE:
        idx = instr.ancestors[1]
        if isinstance(idx, ConstNode):
          idx = idx.val
        else:
          if idx is not None:
            idx = self.global_scope[idx]

        lhs = instr.ancestors[0]
        rhs = self.global_scope[instr.ancestors[2]]

        if isinstance(lhs, GlobalNode):
          buff = self.find_buff(lhs.name)
          if lhs.ptr:
            buff.buff.pointer()[idx*4:idx*4+4].cast('f')[0] = rhs
          else:
            buff.value = rhs
            self.global_scope[lhs] = rhs
        if isinstance(lhs, LocalNode):
          self.global_scope[lhs] = rhs

      elif instr.op is LowIR.OFFSET:
        offset = instr.ancestors[0]
        if offset.op is LowIR.ALU:
          res = self.exec_alu(offset)
          self.global_scope[instr] = res
        elif offset.op is LowIR.LOCAL:
          self.global_scope[instr] = self.global_scope[offset]
        else:
          self.global_scope[instr] = offset.val
      elif instr.op is LowIR.INC:
        loc = instr.ancestors[0]
        self.global_scope[loc]+=1
      else:
        raise TypeError(f"{instr} is not a valid instr")

      self.pc+=1
    return self.pc

  def exec_alu(self, alu_node: ALUNode):
    vals = []
    for operand in alu_node.ancestors:
      if isinstance(operand, ConstNode):
        vals.append(operand.val)
      else:
        vals.append(self.global_scope[operand])
    # chaining multiple binary ops
    if len(vals) > 2 and alu_node.alu in BinaryOps:
      v0 = vals[0]
      for v in vals[1:]:
        res = python_alu[alu_node.alu](v0, v)
        v0 = res
    else:
      res = python_alu[alu_node.alu](*vals)
    return res
  @lru_cache(maxsize=1000)
  def find_buff(self, name):
    for buff in self.buffs['input'] + self.buffs['output']:
      if self.buff2name[buff] == name:
        return buff
    raise KeyError(f"buff with name {name} not found")

class PythonAllocator(Allocator):
  def alloc(self, size:int): return memoryview(bytearray(size))
  def copyin(self, dst, src: memoryview): dst[:] = src[:]
  def copyout(self, dst:memoryview, src): dst[:] = src[:]
  def free(self): return