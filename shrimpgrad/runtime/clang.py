from __future__ import annotations
import ctypes
import subprocess
from typing import List, Tuple
from shrimpgrad.dtype import DType, dtypes
import tempfile
from shrimpgrad.runtime.ops import Op, UnaryOps, BinaryOps, TernaryOps 
from shrimpgrad.runtime.profiler import Profile

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

class ClangProgram:
  def __init__(self):  self.func = {}
  def _dtype_to_c(self, dtype: DType, ptr=True) -> str:
    if dtype == dtypes.float32: return 'float'+ ('*' if ptr else '')
    if dtype == dtypes.int32: return 'int' + ('*' if ptr else '')
    if dtype == dtypes.bool: return 'bool' + ('*' if ptr else '')
  def create_op(self, op: Op, shape:Tuple[int,...], strides:List[int], dtype: DType) -> None:
    ctype = self._dtype_to_c(dtype) if op not in [BinaryOps.XOR, BinaryOps.MOD] else 'int*'
    if op in BinaryOps: self.func[op] = {'args':{'in0':ctype, 'in1':ctype, 'out0': ctype}, 'name':op.name.lower()+'shrimp', 'shape':shape, 'strides':strides}
    if op in UnaryOps: self.func[op] = {'args':{'in0':ctype, 'out0':ctype}, 'name':op.name.lower()+'shrimp', 'shape':shape, 'strides':strides}
  def autogen(self, shape: Tuple[int,...], strides:List[int], dtype: DType):
    for op in c_alu.keys():
      self.create_op(op, shape, strides, dtype)

class ClangCodeGenerator:
  def __init__(self, prg: ClangProgram): self.prg, self._preamble, self._src = prg, '#include<stdio.h>\n#include<math.h>\n', ''
  def render(self):
    for op, opts in self.prg.func.items():
      self._src += self._function(opts['name'], opts['args'], self._loops(opts['shape'], self._op(op, opts['args'].keys(), opts['shape'], opts['strides']))) + '\n'
    return self._preamble + self._src
  # Code Generation (Private)
  def _loop(self,v, s, e, step, body): return f'for(int {v} = {s}; {v} < {e}; {v}+={step}) {{{body}}}'
  def _loops(self, shape: Tuple[int,...], body: str) -> str: 
    def build(shp, depth=0):
      if not shp: return body
      return self._loop(f'i{depth}', 0, shp[0], 1, build(shp[1:], depth+1))
    return build(shape)
  def _loop_vars(self, num_loops): return [f'i{num}' for num in range(num_loops)]
  def _op(self, op: Op, args, shape, strides):
    offset = '+'.join([self._offset(i, strd, 1) for i, strd in zip(self._loop_vars(len(shape)), strides)])
    if op in BinaryOps: c_code = c_alu[op](*self._many_ptrinc(list(args)[0:2], offset))
    if op in UnaryOps: c_code = c_alu[op](self._ptrinc(list(args)[0], offset))
    if op in TernaryOps: c_code = c_alu[op](*self._many_ptrinc(list(args)[0:3], offset))
    return f'out0[{offset}]={c_code};'
  def _ptrinc(self, arg, offset): return f'{arg}[{offset}]' # ptr[offset]
  def _many_ptrinc(self, args, offset): return [self._ptrinc(arg, offset) for arg in args]
  def _offset(self, off:str, strd:int, step:int) -> str: return f'{off}*{strd}*{step}'
  def _function(self, name:str, args: dict, body:str) -> str: return f'void {name} ({self._unpack_args(args)}) {{ { body} }}'
  def _unpack_args(self, args: dict):  return ','.join([f'{typ} {name} ' for name, typ in args.items()])

class ClangCompiler:
  def compile(self, prg: ClangProgram):
    try:
      with tempfile.TemporaryFile() as outfile:
        subprocess.run(['clang', '-include', 'tgmath.h', '-shared', '-march=native', '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-',
                        '-o', str(outfile.name)], check=True, input=ClangCodeGenerator(prg).render().encode('utf-8'))
        return ctypes.CDLL('./'+str(outfile.name))
    except subprocess.CalledProcessError as e:
      print(f"clang failure: {e}") 

class ClangRuntime(metaclass=Profile):
  def __init__(self, lib): self.lib = lib
  def exec(self, op: Op, *args): return getattr(self.lib, op.name.lower() + 'shrimp')(*args)

