from __future__ import annotations
import ctypes, pathlib, tempfile, subprocess
from typing import DefaultDict, List
from shrimpgrad.device import Accelerator, Compiler, ConstBuffer, MallocAllocator, MemBuffer, Renderer, Runtime
from shrimpgrad.dtype import dtypes
from shrimpgrad.engine.lower import ALUNode, ConstNode, GlobalNode, LocalNode, LowIR, LowIRGraph, alu2str
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps

c_alu = {
  UnaryOps.LOG2: lambda x: f'log2({x})',
  UnaryOps.EXP2: lambda x: f'exp2({x})',
  UnaryOps.SQRT: lambda x: f'sqrt({x})', UnaryOps.SIN: lambda x: f'sin({x})',
  UnaryOps.NEG: lambda x: f'-{x}',
  BinaryOps.MUL: lambda x,y: f'{x}*{y}', BinaryOps.ADD: lambda x,y: f'{x}+{y}', BinaryOps.SUB: lambda x,y: f'{x}-{y}', BinaryOps.XOR: lambda x,y: f'{x}^{y}',
  BinaryOps.MAX: lambda x,y: f'fmax({x}, {y})',
  BinaryOps.CMPEQ: lambda x,y: f'{x}=={y}', BinaryOps.CMPLT: lambda x,y: f'{x}<{y}',
  BinaryOps.MOD: lambda x,y: f'{x}%{y}',
  BinaryOps.DIV: lambda x,y: f'{x}/{y}',
  TernaryOps.WHERE: lambda x,y,z: f'{x} ? {y} : {z}'}

class ClangDevice(Accelerator):
  def __init__(self) -> None:
    super().__init__("CLANG", MallocAllocator, ClangRenderer, ClangCompiler, ClangRuntime)
  def allocator(self): return self._allocator()
  def compiler(self): return self._compiler()
  def runtime(self): return self._runtime()
  def renderer(self): return self._renderer()
  def __repr__(self): return "<ClangDevice>"

class ClangRenderer(Renderer):
  def render(self, ir_graph: LowIRGraph):
    return ClangCodeGen([ir_graph]).gen().tostring()

class ClangCodeGen:
  def __init__(self, ir_graphs: List[LowIRGraph]):
    self.preamble ='#include<stdio.h>\n#include<math.h>\n'
    self.gs = []
    self.irgs = ir_graphs
    self.src = []
    self.instr_to_src = {}
    self.indent = 0

  @property
  def spaces(self): return ' ' * self.indent

  def param2src(self, g):
    param = 'float'
    if g[4]== dtypes.int32:
      param = 'int'
    param += '*'
    param += ' ' + g[0]
    return param

  def print(self):
    print(self.preamble)
    print(f"void f_shrimp({','.join([self.param2src(g) for g in self.gs])}) {{")
    for src in self.src:
      print("  " + src)

  def tostring(self):
    name2pos = {g[0]:i for i,g in enumerate(self.gs)}
    params = ','.join([self.param2src(g) for g in self.gs])
    out = f"{self.preamble}"
    out += (f"void f_shrimp({params}) {{\n")
    for src in self.src:
      out += "  " + src + "\n"
    out += '}'
    return out, name2pos

  def gen(self):
    for irg in self.irgs:
      instrs = irg.G
      i = 0
      while i < len(instrs):
        instr = instrs[i]
        if instr.op is LowIR.END_LOOP:
          self.indent -= 2
          self.src.append(self.spaces+'}')
          i += 1
          continue
        if instr.op is LowIR.CONST:
          i += 1
          continue
        elif instr.op is LowIR.LOCAL:
          self.instr_to_src[instr] = instr.name
          if instr.ancestors[0].op is LowIR.ALU:
            alu_node = instr.ancestors[0]
            rhs = self.instr_to_src[alu_node]
            self.src.append(f"{self.spaces}int {instr.name} = {rhs};")
          else: # const
            self.src.append(f"{self.spaces}int {instr.name} = {instr.ancestors[0].val};")
        elif instr.op is LowIR.GLOBAL:
          self.gs.append((instr.name, instr.ptr, instr.pos, instr.mutable, instr.dtype))
        elif instr.op is LowIR.BEGIN_LOOP:
          s, e = instr.ancestors[0], instr.ancestors[1].val
          self.src.append(f"{self.spaces}for (; {s.name} < {e}; {s.name}++) {{ ")
          self.indent += 2
        elif instr.op is LowIR.ADDRESS:
          addr = ''
          for idx, stride in zip(instr.idx, instr.stride):
            val = idx.name if isinstance(idx, LocalNode) else idx
            addr += f"{val}*{stride}+"
          self.instr_to_src[instr] = addr[:-1] if addr else 0
        elif instr.op is LowIR.LOAD:
          g = instr.ancestors[0]
          # Ensure the global is defined. Since all LowIRGraph's share the same symbol table
          # sometimes a later IR won't define a global already defined in a previous IR.
          # Here we define it to ensure it ends up in the parameter list.
          is_defined = any(g_[0] == g.name for g_ in self.gs)
          if not is_defined: self.gs.append((g.name, g.ptr, g.pos, g.mutable, g.dtype))
          addr = instr.ancestors[1]
          if addr is not None:
            idx = self.instr_to_src[addr] if not isinstance(addr, ConstNode) else addr.val
            self.instr_to_src[instr] = f"{g.name}[{idx}]"
          else:
            self.instr_to_src[instr] = "(*"+g.name+")"
        elif instr.op is LowIR.ALU:
          self.exec_alu(instr)
        elif instr.op is LowIR.STORE: self.store(instr)
        elif instr.op is LowIR.OFFSET: self.offset(instr)
        elif instr.op is LowIR.INC:
          loc = instr.ancestors[0]
          self.src.append(f"{self.spaces}{loc.name}++;")
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
        r = f"*{lhs.name} = {rhs}"
    else:
      r = f"*{lhs.name} = {rhs}"
    self.src.append(self.spaces + r + ";")
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
    self.instr_to_src[alu_node] = c_alu[alu_node.alu](*vals)

class ClangCompiler(Compiler):
  def compile(self, src: str) -> bytes:
    with tempfile.NamedTemporaryFile(delete=True) as outfile:
      subprocess.check_output(['clang', '-include', 'tgmath.h', '-shared', '-march=native', '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-',
                      '-o', str(outfile.name)], input=src.encode('utf-8'))
      return pathlib.Path(outfile.name).read_bytes()

class ClangRuntime(Runtime):
  def _exec(self, *args):
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(self.lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))['f_shrimp']
      self.fxn(*args)

  def exec(self, lib: bytes, buffs: DefaultDict[str, List[MemBuffer | ConstBuffer]], buff2name, name2pos):
    self.lib = lib
    self.buffs = buffs
    self.buff2name = buff2name
    vin = [None]*len(name2pos)

    # TODO: Remove this hacky bit used to extract
    # the result of an output of const by making consts use buffers of size 1
    const_vals = []

    for buff in buffs['output']:
      name = buff2name[buff]
      if buff.__class__ is MemBuffer:
        vin[name2pos[name]] = (ctypes.byref(buff.buff._pointer(ctypes.c_float)))
      else:
        vin[name2pos[name]] = ctypes.byref(val:=ctypes.c_float(buff.value))
        const_vals.append((buff,val))

    for buff in buffs['input']:
      name = buff2name[buff]
      if buff.__class__ is MemBuffer:
        vin[name2pos[name]] = (ctypes.byref(buff.buff._pointer(ctypes.c_float)))
      else:
        vin[name2pos[name]] = ctypes.byref(ctypes.c_float(buff.value))

    self._exec(*vin)
    for buff, val in const_vals:
      buff.value = val.value
