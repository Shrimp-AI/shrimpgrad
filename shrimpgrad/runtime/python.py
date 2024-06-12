from __future__ import annotations
import base64
import ctypes
import math
import operator
import pickle
from typing import DefaultDict, List
from shrimpgrad.device import Accelerator, Allocator, Compiler, ConstBuffer, MemBuffer, Runtime
from shrimpgrad.engine.lower import ALUNode, AddressNode, BeginLoopNode, ConstNode, EndLoopNode, GlobalNode, LoadNode, LocalNode, LowIRGraph, Node, StoreNode
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps
from collections import deque

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
    return f"<PythonDevice>"

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

  def _loop(self, s,e,instrs):
    start_loop_pc = self.pc
    for i in range(s,e):
      # Go to instruction after the for stmt
      self.pc = start_loop_pc + 1
      print(f"Loop iter= {i} pc={self.pc}")
      end_loop_pc = self._exec(instrs)
    return end_loop_pc

  def exec(self, lib: bytes, buffs: DefaultDict[str, List[MemBuffer | ConstBuffer]], buff2name):
    ir_graph: LowIRGraph = pickle.loads(lib)
    self.buffs = buffs
    self.buff2name = buff2name
    self._exec(ir_graph.G)

  def _exec(self, instrs: List[Node]):
    while self.pc < len(instrs):
      instr = instrs[self.pc]
      print(f"PC={self.pc} instr={instr}")
      if isinstance(instr, EndLoopNode):
        print("END LOOP jump")
        return self.pc
      if isinstance(instr, ConstNode):
        self.pc += 1
        continue
      if isinstance(instr, LocalNode):
        if isinstance(instr.ancestors[0], ALUNode):
          alu_node = instr.ancestors[0]
          operands = alu_node.ancestors
          self.gobal_scope[instr] = python_alu[alu_node.alu](*[self.global_scope[o] for o in operands])
        else:
          # const
          self.global_scope[instr] = instr.ancestors[0].val
      if isinstance(instr, GlobalNode):
        self.global_scope[instr] = self.find_buff(instr.name).buff
      if isinstance(instr, BeginLoopNode):
        s, e = instr.ancestors[0].ancestors[0].val, instr.ancestors[1].val
        print(f"loop start={s} end={e}")
        end_loop_pc = self._loop(s, e, instrs)
        self.pc = end_loop_pc
      if isinstance(instr, AddressNode):
        idxs = instr.idx
        strides = instr.stride
        addr = 0
        for idx, stride in zip(idxs, strides):
          i = idx
          if isinstance(idx, LocalNode):
            i = self.global_scope[idx]
          addr += i * stride
        self.address = addr
      if isinstance(instr, LoadNode):
        node = instr.ancestors[0]
        if isinstance(node, GlobalNode):
          buff = self.find_buff(node.name)
          if node.ptr:
            loaded = buff.buff.pointer(ctypes.c_float)[self.address]
          else:
            loaded = buff.value
          self.load_stack.append(loaded)
      if isinstance(instr, ALUNode):
        vals = []
        for operand in instr.ancestors:
          if isinstance(operand, LoadNode):
            vals.append(self.load_stack.popleft())
          elif isinstance(operand, LocalNode):
            vals.append(self.global_scope[operand])
          elif isinstance(operand, ConstNode):
            vals.append(operand.val)
          else:
            raise ValueError(f"operand {operand} is not a valid value")
        res = python_alu[instr.alu](*vals)
        self.global_scope[instr] = res
      if isinstance(instr, StoreNode):
        pass


      self.pc+=1

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