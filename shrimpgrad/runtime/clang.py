from __future__ import annotations
import ctypes
from enum import Enum, auto
import subprocess
from typing import Type, Union
from shrimpgrad.dtype import DType

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
  def pointer(self):
    return ctypes.cast(self._data.tobytes(), ctypes.POINTER(ctypes.c_byte))
  def copyin(self, src: memoryview): 
    self.allocator.copyin(self._data, src)
  def copyout(self, dst: memoryview):
    self.allocator.copyout(dst, self._data)
  def view(self):
    return Buffer(self.allocator, self.size, self.dtype) 

class ClangOps:
  pass
  
class ClangCompiler:
  @staticmethod
  def compile(src: ClangProgram):
    try:
      # Invoke the C compiler (gcc) using subprocess
      src.tofile('tmp.c')
      subprocess.run(['clang', '-include', 'tgmath.h', '-shared', '-march=native', '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-',
                              'tmp.c', '-o', 'cshrimp.so'], check=True)
      return ctypes.CDLL('./cshrimp.so')
    except subprocess.CalledProcessError as e:
      print(f"Compilation failed: {e}") 

class ClangProgram:
  def __init__(self): 
    self._src = '#include<stdio.h>\n void add(float *x, float *y, float *out) { *out = *x + *y; }'
  def tostring(self): return self._src
  def tofile(self, filepath: str):
    with open(filepath, "w", buffering=1024*1024) as f:
      f.write(self._src)

class ClangRuntime:
  def __init__(self, lib):
    self.lib = lib
  def exec(self, op, *args):
    if op == BinaryOps.ADD:
      self.lib.add.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
      return self.lib.add(*args)