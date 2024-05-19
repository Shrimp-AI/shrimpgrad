from __future__ import annotations
import ctypes
import subprocess

from shrimpgrad.dtype import DType

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
  def compile(src: ClangProgram, fout:str):
    try:
      # Invoke the C compiler (gcc) using subprocess
      src.tofile('tmp.c')
      subprocess.run(['clang', '-include', 'tgmath.h', '-march=native', '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-',
                              'tmp.c', '-o', 'cshrimp'], check=True)
      return 'cshrimp'
    except subprocess.CalledProcessError as e:
      print(f"Compilation failed: {e}") 

class ClangProgram:
  def __init__(self): 
    self._src = '#include<stdio.h>\n int add(int x, int y) { return x + y; } int main() { int r = add(1,2); printf("%d\\n", r); return 0; }'
  def tostring(self): return self._src
  def tofile(self, filepath: str):
    with open(filepath, "w", buffering=1024*1024) as f:
      f.write(self._src)

class ClangRuntime:
  @staticmethod
  def run(elf: str):
    try: subprocess.run(['./cshrimp'], check=True)
    except subprocess.CalledProcessError as e:
      print(f"Error running C program: {e}")