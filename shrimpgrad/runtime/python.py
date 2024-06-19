from __future__ import annotations
import base64
import ctypes
import pickle
from typing import DefaultDict, List
from shrimpgrad.device import Accelerator, Allocator, Compiler, ConstBuffer, MemBuffer, Runtime
from shrimpgrad.engine.lower import ALUNode, ConstNode, GlobalNode, LocalNode, LowIR, LowIRGraph, alu2str
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps

pyalu2src = {
  UnaryOps.LOG2: lambda x: f"math.log2({x}) if {x} > 0 else -math.inf if {x} == 0 else math.nan",
  UnaryOps.EXP2: lambda x: f"math.exp({x}*math.log(2))",
  UnaryOps.SQRT: lambda x: f"math.sqrt({x}) if {x} >= 0 else math.nan", UnaryOps.SIN: lambda x: f"math.sin({x})",
  UnaryOps.NEG: lambda x: f"(not {x}) if isinstance({x}, bool) else -{x}",
  BinaryOps.MUL: lambda x, y: f"operator.mul({x},{y})", BinaryOps.ADD: lambda x,y: f"operator.add({x},{y})", BinaryOps.SUB: lambda x,y: f"operator.sub({x},{y})", BinaryOps.XOR: lambda x,y: f"operator.xor({x},{y})",
  BinaryOps.MAX: lambda x,y: f"max({x},{y})", BinaryOps.CMPEQ: lambda x,y: f"operator.eq({x},{y})", BinaryOps.CMPLT: lambda x,y: f"operator.lt({x},{y})",
  BinaryOps.MOD: lambda x,y: f"abs(int({x}))%abs(int({y}))*(1,-1)[{x}<0]",
  BinaryOps.DIV: lambda x,y: f"int({x}/{y}) if isinstance({x}, int) else ({x}/{y} if {y} != 0 else {x}*math.inf)",
  TernaryOps.WHERE: lambda x,y,z: f"{y} if {x} else {z}"
}

class PythonDevice(Accelerator):
  def __init__(self):
    super().__init__("PYTHON", PythonAllocator, PythonRenderer, PythonCompiler, PythonRuntime)
  def compiler(self) -> PythonCompiler: return self._compiler()
  def allocator(self) -> PythonAllocator: return self._allocator()
  def runtime(self) -> PythonRuntime: return self._runtime()
  def renderer(self) -> PythonRenderer: return self._renderer()
  def __repr__(self) -> str:
    return "<PythonDevice>"

class PythonAllocator(Allocator):
  def alloc(self, size:int): return memoryview(bytearray(size))
  def copyin(self, dst, src: memoryview): dst[:] = src[:]
  def copyout(self, dst:memoryview, src): dst[:] = src[:]
  def free(self): return

class PythonRenderer:
  def render(self, ir_graph: LowIRGraph) -> str: return base64.b64encode(pickle.dumps(PyCodeGen([ir_graph]).gen().tostring()))

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonRuntime(Runtime):
  def exec(self, lib: bytes, buffs: DefaultDict[str, List[MemBuffer | ConstBuffer]], buff2name):
    src = pickle.loads(lib)
    self.buffs = buffs
    self.buff2name = buff2name
    vin = {}
    for buff in buffs['input'] + buffs['output']:
      if isinstance(buff, MemBuffer):
        vin[buff2name[buff]] = buff.buff._pointer(ctypes.c_float)
      else:
        vin[buff2name[buff]] = buff
    exec(src, vin) # pylint: disable=exec-used

class PyCodeGen:
  def __init__(self, ir_graphs: List[LowIRGraph]):
    self.irgs = ir_graphs
    self.gs = []
    self.preamble = "import operator\nimport math\n"
    self.src = []
    self.instr_to_src = {}
    self.indent = 0

  @property
  def spaces(self): return ' ' * self.indent

  def print(self):
    print(self.preamble)
    print(f"def f_{id(self)}({','.join([g[0] for g in self.gs])}):")
    for src in self.src:
      print("  " + src)

  def tostring(self):
    params = ','.join([g[0] for g in self.gs])
    out = f"{self.preamble}"
    out += f"def f_{id(self)}({params}):\n"
    for src in self.src:
      out += "  " + src + "\n"
    out += f"f_{id(self)}({params})"
    return out

  def gen(self):
    for irg in self.irgs:
      instrs = irg.G
      i = 0
      while i < len(instrs):
        instr = instrs[i]
        if instr.op is LowIR.END_LOOP:
          i+=1
          self.indent -= 2
          continue
        if instr.op is LowIR.CONST:
          i+=1
          continue
        elif instr.op is LowIR.LOCAL:
          self.instr_to_src[instr] = instr.name
          if instr.ancestors[0].op is LowIR.ALU:
            alu_node = instr.ancestors[0]
            rhs = self.instr_to_src[alu_node]
            self.src.append(f"{self.spaces}{instr.name} = {rhs}")
          else:
            self.src.append(f"{self.spaces}{instr.name} = {instr.ancestors[0].val}")
        elif instr.op is LowIR.GLOBAL:
          self.gs.append((instr.name, instr.ptr, instr.pos, instr.mutable))
        elif instr.op is LowIR.BEGIN_LOOP:
          s, e = instr.ancestors[0], instr.ancestors[1].val
          self.src.append(f"{self.spaces}for {s.name} in range({s.ancestors[0].val},{e}):")
          self.indent += 2
        elif instr.op is LowIR.ADDRESS:
          addr = ''
          for idx, stride in zip(instr.idx, instr.stride):
            val = idx.name if isinstance(idx, LocalNode) else idx
            addr += f"{val}*{stride}+"
          self.instr_to_src[instr] = addr[:-1] if addr else 0
        elif instr.op is LowIR.LOAD:
          g = instr.ancestors[0]
          addr = instr.ancestors[1]
          if addr is not None:
            idx = self.instr_to_src[addr] if not isinstance(addr, ConstNode) else addr.val
            self.instr_to_src[instr] = f"{g.name}[{idx}]"
          else:
            self.instr_to_src[instr] = g.name + '.value'
        elif instr.op is LowIR.ALU:
          self.exec_alu(instr)
        elif instr.op is LowIR.STORE: self.store(instr)
        elif instr.op is LowIR.OFFSET: self.offset(instr)
        elif instr.op is LowIR.INC:
          loc = instr.ancestors[0]
          self.src.append(f"{self.spaces}{loc.name}+=1")
        else:
          raise TypeError(f"{instr} is not a valid instr")
        i+=1
    return self

  def store(self, instr):
    idx = instr.ancestors[1]
    if isinstance(idx, ConstNode):
      idx = idx.val
    else:
      if idx is not None:
        idx = self.instr_to_src[idx]

    lhs = instr.ancestors[0]
    rhs = self.instr_to_src[instr.ancestors[2]]

    if isinstance(lhs, GlobalNode):
      if lhs.ptr:
        r = f"{lhs.name}[{idx}] = {rhs}"
      else:
        r = f"{lhs.name}.value = {rhs}"
    else:
      r = f"{lhs.name} = {rhs}"
    self.src.append(self.spaces + r)
    return r

  def offset(self, instr):
    offset = instr.ancestors[0]
    if offset.op is LowIR.ALU:
      self.instr_to_src[instr] = self.instr_to_src[offset]
    elif offset.op is LowIR.LOCAL:
      self.instr_to_src[instr] = offset.name
    else:
      self.instr_to_src[instr] = offset.val

  def exec_alu(self, alu_node: ALUNode):
    vals = []
    for operand in alu_node.ancestors:
      if isinstance(operand, ConstNode):
        vals.append(operand.val)
      elif operand.op is LowIR.LOCAL:
        vals.append(operand.name)
      else:
        vals.append(self.instr_to_src[operand])
    if len(vals) > 2:
      op = alu2str(alu_node.alu)
      code = ''
      for i in range(len(vals)):
        code += f"{vals[i]} {op} "
      self.instr_to_src[alu_node] = code[:-2]
      return
    self.instr_to_src[alu_node] = pyalu2src[alu_node.alu](*vals)