from __future__ import annotations
import ctypes
from enum import Enum, auto
import subprocess
from typing import List, Tuple, Type, Union
from shrimpgrad.dtype import DType, dtypes
import tempfile

class UnaryOps(Enum): EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum):
  ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPEQ = auto(); XOR = auto() # noqa: E702
class TernaryOps(Enum): WHERE = auto(); MULACC = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class BufferOps(Enum): LOAD = auto(); CONST = auto(); STORE = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); ASSIGN = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[LoadOps], Type[TernaryOps], Type[BufferOps]]

class Allocator:
  def alloc(self): raise NotImplementedError('implement alloc')
  def free(self): raise NotImplementedError('implement free')
  def copyin(self): raise NotImplementedError('implement copyin')
  def copyout(self): raise NotImplementedError('implement copyout')

class MallocAllocator(Allocator):
  def alloc(self, size:int):
    return (ctypes.c_uint8 * size)()
  def copyin(self, dst, src:memoryview):
    ctypes.memmove(dst, src, len(src))
  def copyout(self, dst:memoryview, src):
    ctypes.memmove(dst, src, len(dst))
  def free(self): return

class Buffer:
  def __init__(self, allocator: Allocator, size:int, dtype: DType):
    self.allocator, self.dtype, self.size = allocator, dtype, size
    self._ref_count = 1
    self._data = memoryview(allocator.alloc(self.dtype.bytes * self.size))
  def pointer(self, to_type=ctypes.c_byte):
    return ctypes.cast(ctypes.addressof(to_type.from_buffer(self._data)), ctypes.POINTER(to_type*ctypes.sizeof(to_type))).contents
  def copyin(self, src: memoryview): 
    self.allocator.copyin(self.pointer(), src)
  def copyout(self, dst: memoryview):
    self.allocator.copyout(dst, self.pointer())
  def view(self):
    return Buffer(self.allocator, self.size, self.dtype) 

c_alu = {
  UnaryOps.LOG2: 'log2',
  UnaryOps.EXP2: 'exp2',
  UnaryOps.SQRT: 'sqrt', UnaryOps.SIN: 'sin',
  # UnaryOps.NEG: '-',
  BinaryOps.MUL: '*', BinaryOps.ADD: '+', BinaryOps.SUB: '-', BinaryOps.XOR: '^',
  # TODO: rendering binary op functions fix needed BinaryOps.MAX: 'MAX'
  BinaryOps.CMPEQ: '==', BinaryOps.CMPLT: '<',
  BinaryOps.MOD: '%',
  BinaryOps.DIV: '/',
  TernaryOps.WHERE: 'if else'}

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

class ClangRenderer:
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
  def _op(self, op: Op, in_bufs, shape, strides):
    offset = '+'.join([self._offset(i, strd, 1) for i, strd in zip(self._loop_vars(len(shape)), strides)])
    if op in BinaryOps:
      c_code = c_alu[op].join([f'{in_b}[{offset}]' for in_b in list(in_bufs)[0:2]]) + ';'
      return f'out0[{offset}]={c_code}'
    if op in UnaryOps:
      return f'out0[{offset}]={c_alu[op]}({list(in_bufs)[0]}[{offset}])' + ';' 

  def _offset(self, off:str, strd:int, step:int) -> str: return f'{off}*{strd}*{step}'
  def _function(self, name:str, args: dict, body:str) -> str: return f'void {name} ({self._unpack_args(args)}) {{ { body} }}'
  def _unpack_args(self, args: dict):  return ','.join([f'{typ} {name} ' for name, typ in args.items()])

class ClangCompiler:
  @staticmethod
  def compile(prg: ClangProgram):
    try:
      with tempfile.TemporaryFile() as outfile:
        subprocess.run(['clang', '-include', 'tgmath.h', '-shared', '-march=native', '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-',
                        '-o', str(outfile.name)], check=True, input=ClangRenderer(prg).render().encode('utf-8'))
        return ctypes.CDLL('./'+str(outfile.name))
    except subprocess.CalledProcessError as e:
      print(f"clang failure: {e}") 

class ClangRuntime:
  def __init__(self, lib): self.lib = lib
  def exec(self, op, *args):
    if op == BinaryOps.ADD:
      self.lib.addshrimp.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
      return self.lib.addshrimp(*args)
    if op == BinaryOps.MUL:
      self.lib.mulshrimp.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
      return self.lib.mulshrimp(*args)