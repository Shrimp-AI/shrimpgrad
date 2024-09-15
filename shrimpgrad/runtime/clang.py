from __future__ import annotations
import math
import ctypes, pathlib, tempfile, subprocess
from typing import DefaultDict, Dict, List, Optional, Tuple, cast
from shrimpgrad.device import Buffer, Device, Compiler, Jitable, MallocAllocator, MemBuffer, Renderer, Runtime
from shrimpgrad.dtype import DType, dtypes, to_ctype
from shrimpgrad.engine.lower import ALUNode, AddressNode, BeginLoopNode, ConstNode, GlobalNode, LocalNode, LowIR, LowIRGraph, alu2str
from shrimpgrad.engine.runner import CompiledKernel
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps
from shrimpgrad.symbolic import Expr, render

c_alu = {
  UnaryOps.LOG2: lambda x: f'log2({x})',
  UnaryOps.EXP2: lambda x: f'exp2({x})',
  UnaryOps.SQRT: lambda x: f'sqrt({x})', UnaryOps.SIN: lambda x: f'sin({x})',
  UnaryOps.NEG: lambda x: f'-{x}',
  BinaryOps.MUL: lambda x,y: f'{x}*{y}', BinaryOps.ADD: lambda x,y: f'{x}+{y}', BinaryOps.SUB: lambda x,y: f'{x}-{y}', BinaryOps.XOR: lambda x,y: f'{x}^{y}',
  BinaryOps.MAX: lambda x,y: f"(({x}>{y})?{x}:{y})",
  BinaryOps.CMPEQ: lambda x,y: f'{x}=={y}', BinaryOps.CMPLT: lambda x,y: f'{x}<{y}',
  BinaryOps.MOD: lambda x,y: f'{x}%{y}',
  BinaryOps.DIV: lambda x,y: f'{x}/{y}',
  TernaryOps.WHERE: lambda x,y,z: f'{x} ? {y} : {z}'}

def render_type(dtype: DType) -> str:
  if dtype == dtypes.float32: return 'float'
  if dtype == dtypes.int32: return 'int'
  if dtype == dtypes.bool_: return '_Bool'
  if dtype == dtypes.uint8: return 'unsigned char'
  raise TypeError(f"dtype {dtype} is not supported.")

class ClangDevice(Device, Jitable):
  def __init__(self) -> None:
    super().__init__("CLANG", MallocAllocator, ClangRenderer, ClangCompiler, ClangRuntime)
    self.compiler_ = self._compiler()
  def allocator(self): return self._allocator()
  def compiler(self): return self.compiler_
  def runtime(self): return self._runtime()
  def renderer(self): return self._renderer()
  def jitify(self, kernels: List[CompiledKernel], input_buffers: List[Buffer]):
    native_src = []
    prgs: List[ClangProgram] = [ck.prg for ck in kernels]
    buffs = [ck.buffs for ck in kernels]
    buff2names = [ck.buff2name for ck in kernels]
    func_sigs = set()
    # Add all imports
    native_src.append(prgs[0].preamble)
    # Add all the functions
    for prg in prgs:
      if prg.fsig in func_sigs:
        continue
      func_sigs.add(prg.fsig)
      fsrc = f"{prg.fsig} {{\n{prg.fbdy}}}\n"
      native_src.append(fsrc)
    args = [f"{render_type(input_buffers[i].dtype)}* arg{i}" for i in range(len(input_buffers))]
    body = {}
    native_src.append(f"\nvoid batched({','.join(args)}) {{ \n")
    in_names = {}
    declared = set()
    for i,buff in enumerate(buffs):
      for buff_ in buff['input'] + buff['output']:
        assert isinstance(buff_, MemBuffer), f"invalid buffer in clang.jitify {buff_}"
        real_buff = buff_.buff
        if real_buff not in input_buffers:
          if (name:=f"{buff2names[i][buff_]}{i}") in declared: continue
          declared.add(name)
          body[name] = (f"  {render_type(real_buff.dtype)}* {name} = ({render_type(real_buff.dtype)}*)0x{ctypes.addressof(real_buff._pointer(to_ctype(real_buff.dtype))):X};\n")
        else:
          in_names[buff2names[i][buff_]] = input_buffers.index(real_buff)
 
    for i,prg in enumerate(prgs):
      args = []
      for n, _ in prg.args2pos.items():
        if n in in_names:
          args.append(f"arg{in_names[n]}")
          continue
        name = f"{n}{i}"
        if name in body:
          native_src.append(body[f"{n}{i}"])
          del body[name]
        args.append(f"{n}{i}")
      native_src.append(f"  {prg.fname}({','.join(args)});\n")
    native_src.append("}\n")
    native_src_ = "".join(native_src)
    native_lib = self.compiler().cached_compile(native_src_)
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(native_lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))['batched']
      return self.fxn

class ClangProgram:
  def __init__(self, src:str, preamble: str, fsig: str, fbdy: str,
               fname: str, args2pos: Dict[str, int]):
    self.preamble, self.fsig, self.fbdy, self.fname, self.args2pos, self.src = preamble, fsig, fbdy, fname, args2pos, src

class ClangRenderer(Renderer):
  def render(self, ir: LowIRGraph, name=None):
    return ClangCodeGen([ir], name).gen().to_program()

class ClangCodeGen:
  def __init__(self, ir_graphs: List[LowIRGraph], func_name=None):
    self.preamble ='#include<stdio.h>\n#include<math.h>\n'
    self.func_name = func_name
    self.gs = []
    self.irgs = ir_graphs
    self.src = []
    self.instr_to_src = {}
    self.indent = 0

  @property
  def spaces(self): return ' ' * self.indent

  def param2src(self, g):
    param = render_type(g[4]) 
    param += '*'
    param += ' ' + g[0]
    return param

  def _name(self):
    return f"f_{id(self.irgs[0])}" if self.func_name is None else self.func_name

  def to_program(self):
    params = ','.join([self.param2src(g) for g in self.gs])
    src_ = self.preamble
    fname = self._name()
    fsig = f"void {fname}({params})"
    src_ += fsig + " { \n"
    body = ''
    for src in self.src:
      body += "  " + src + "\n"
    src_ += body + "}\n"
    return ClangProgram(src_, self.preamble, fsig, body, fname, {g[0]:i for i,g in enumerate(self.gs)})

  def print(self): print(self.to_program().src)

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
          assert isinstance(instr, ConstNode), f'invalid const instruction {instr}'
          if isinstance(instr.val, bool):
            self.instr_to_src[instr] = 1.0 if instr.val else 0.0
          elif math.isinf(instr.val):
            self.instr_to_src[instr] = '-INFINITY' if instr.val <0 else 'INFINITY' 
          else: self.instr_to_src[instr] = instr.val
          continue
        elif instr.op is LowIR.LOCAL:
          assert isinstance(instr, LocalNode)
          self.instr_to_src[instr] = instr.name
          typ = render_type(instr.dtype)
          if instr.ancestors[0].op is LowIR.ALU:
            alu_node = instr.ancestors[0]
            rhs = self.instr_to_src[alu_node]
            self.src.append(f"{self.spaces}{typ} {instr.name} = {rhs};")
          else: # const
            assert isinstance(const_node:=instr.ancestors[0], ConstNode)
            rhs = self.instr_to_src[const_node]
            self.src.append(f"{self.spaces}{typ} {instr.name} = {rhs};")
        elif instr.op is LowIR.GLOBAL:
          assert isinstance(instr, GlobalNode)
          self.gs.append((instr.name, instr.ptr, instr.pos, instr.mutable, instr.dtype))
        elif instr.op is LowIR.BEGIN_LOOP:
          assert isinstance(instr, BeginLoopNode), "invalid instr for node type"
          assert isinstance(instr.ancestors[0], LocalNode), "invalid loop variable" 
          assert isinstance(instr.ancestors[1], ConstNode), "invalid loop bound" 
          s, e = instr.ancestors[0], instr.ancestors[1].val
          self.src.append(f"{self.spaces}for (; {s.name} < {e}; {s.name}++) {{ ")
          self.indent += 2
        elif instr.op is LowIR.ADDRESS:
          instr = cast(AddressNode, instr)
          if instr.vt is not None:
            exprs = instr.vt.to_symbolic([ix.name for ix in instr.idx if isinstance(ix, LocalNode)]) 
            self.instr_to_src[instr] = exprs 
            i+=1
            continue
          addr = ''
          for idx, stride in zip(instr.idx, instr.stride):
            val = idx.name if isinstance(idx, LocalNode) else idx
            addr += f"{val}*{stride}+"
          self.instr_to_src[instr] = addr[:-1] if addr else 0
        elif instr.op is LowIR.LOAD: self.load(instr)
        elif instr.op is LowIR.ALU: self.exec_alu(instr)
        elif instr.op is LowIR.STORE: self.store(instr)
        elif instr.op is LowIR.OFFSET: self.offset(instr)
        elif instr.op is LowIR.INC:
          loc = instr.ancestors[0]
          self.src.append(f"{self.spaces}{loc.name}++;")
        else:
          raise TypeError(f"{instr} is not a valid instr")
        i+=1
    return self
  
  def load(self, instr):
    g = instr.ancestors[0]
    # Ensure the global is defined. Since all LowIRGraph's share the same symbol table
    # sometimes a later IR won't define a global already defined in a previous IR.
    # Here we define it to ensure it ends up in the parameter list.
    if g.__class__ is GlobalNode:
      is_defined = any(g_[0] == g.name for g_ in self.gs)
      if not is_defined: self.gs.append((g.name, g.ptr, g.pos, g.mutable, g.dtype))
      addr = instr.ancestors[1]
      if addr is not None:
        idx = self.instr_to_src[addr] if not isinstance(addr, ConstNode) else addr.val
        if isinstance(idx, Tuple):
          idx = cast(Tuple[Expr,Optional[Expr]],idx)
          iexpr, bexpr = render(idx[0]), render(idx[1]) if idx[1] is not None else None
          if bexpr is not None:
            self.instr_to_src[instr] = f"({bexpr}) ? {g.name}[{iexpr}] : 0.0f"
          else:
            self.instr_to_src[instr] = f"{g.name}[{iexpr}]"
          return
        idx = self.instr_to_src[addr] if not isinstance(addr, ConstNode) else addr.val
        self.instr_to_src[instr] = f"{g.name}[{idx}]"
      else:
        self.instr_to_src[instr] = "(*"+g.name+")"
    else:
      self.instr_to_src[instr] = "("+g.name+")"

  def store(self, instr):
    bexpr = None
    idx = instr.ancestors[1]
    if isinstance(idx, ConstNode):
      idx = idx.val
    else:
      if idx is not None:
        idx = self.instr_to_src[idx]
        if isinstance(idx, Tuple):
          idx = cast(Tuple[Expr,Optional[Expr]],idx)
          iexpr, bexpr = render(idx[0]), render(idx[1]) if idx[1] is not None else None
          idx = iexpr

    lhs = instr.ancestors[0]
    rhs = self.instr_to_src[instr.ancestors[2]]

    if isinstance(lhs, GlobalNode):
      if lhs.ptr:
        r = f"{lhs.name}[{idx if idx is not None else 0}] = {rhs}"
        if bexpr is not None:
          r = f"if (!({bexpr})) {{ {r}; }}"
      else:
        r = f"*{lhs.name} = {rhs}"
    else:
      r = f"{lhs.name} = {rhs}"
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
    if len(vals) > 2 and alu_node.alu not in TernaryOps:
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
      subprocess.check_output(['clang', '-include', 'tgmath.h', '-shared',
                               '-march=native', '-O2', '-Wall', '-Werror',
                               '-ftree-vectorize', '-x', 'c',
                               '-fPIC', '-', '-o', outfile.name], input=src.encode('utf-8'))
      return pathlib.Path(outfile.name).read_bytes()

class ClangRuntime(Runtime):
  def _exec(self, func_name:str, *args):
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(self.lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[func_name]
      self.fxn(*args)
  
  def batched_exec(self, lib: bytes, func_names, buffs, buff2name, name2pos):
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      slib = ctypes.CDLL(str(cached_file_path.name))
      for i, func_name in enumerate(func_names):
        vin = [None]*len(name2pos[i])
        for buff in buffs[i]['output']:
          name = buff2name[buff]
          if name not in name2pos[i]: continue
          vin[name2pos[i][name]] = (ctypes.byref(buff.buff._pointer(to_ctype(buff.buff.dtype))))

        for buff in buffs[i]['input']:
          name = buff2name[buff]
          if name not in name2pos[i]: continue
          vin[name2pos[i][name]] = (ctypes.byref(buff.buff._pointer(to_ctype(buff.buff.dtype))))
        slib[func_name](*vin)

  def exec(self, lib: bytes, func_name:str, buffs: DefaultDict[str, List[MemBuffer]], buff2name, name2pos):
    self.lib = lib
    self.buffs = buffs
    self.buff2name = buff2name
    vin = [None]*len(name2pos)

    for buff in buffs['output']:
      name = buff2name[buff]
      if name not in name2pos: continue
      vin[name2pos[name]] = (ctypes.byref(buff.buff._pointer(to_ctype(buff.buff.dtype))))

    for buff in buffs['input']:
      name = buff2name[buff]
      if name not in name2pos: continue
      vin[name2pos[name]] = (ctypes.byref(buff.buff._pointer(to_ctype(buff.buff.dtype))))

    self._exec(func_name, *vin)
