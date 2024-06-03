
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from typing import Type
from shrimpgrad.dtype import ConstType, DType
from shrimpgrad.meta.singleton import Singleton
from shrimpgrad.runtime.clang import ClangCompiler, ClangRuntime
from shrimpgrad.view import ViewTracker

class Device(metaclass=Singleton):
  def __init__(self, name:str): self.name = name
  def __eq__(self, other: Device): return self.name == other.name
  def __hash__(self): return id(self)

class Compiler:
  def compile(self): raise NotImplementedError('implement compile') 

class Runtime:
  def exec(self): raise NotImplementedError('implement exec') 

class Accelerator(Device):
  def __init__(self, name:str, allocator: Type[Allocator], compiler: Type[Compiler], runtime: Type[Runtime]) -> None:
    super().__init__(name)
    self._allocator, self._compiler, self._runtime = allocator, compiler, runtime
  def compiler(self): raise NotImplementedError('implement compiler for accelerator')
  def allocator(self): raise NotImplementedError('implement allocator for accelerator')
  def runtime(self): raise NotImplementedError('implement runtime for accelerator')

class HostDevice(Device): 
  def copyto(self, accelerator: Accelerator): raise NotImplementedError('implement copyto')

class CPU(HostDevice):
  def __init__(self):
    super().__init__("CPU") 
  def copyto(self, accelerator: Accelerator): pass

class ClangDevice(Accelerator):
  def __init__(self) -> None:
    super().__init__("CLANG", MallocAllocator, ClangCompiler, ClangRuntime)
  
  def allocator(self):
    return self._allocator()

  def compiler(self):
    return self._compiler()

  def runtime(self):
    return self._runtime()

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
  def __init__(self, device: Device, size:int, dtype: DType):
    self.allocator, self.dtype, self.size = device.allocator() if isinstance(device, Accelerator) else None, dtype, size
    self._ref_count = 1
  @property
  def allocated(self): return hasattr(self, '_buf')
  @property
  def nbytes(self): return self.size * self.dtype.bytes
  def allocate(self, with_data=None):
    if with_data is None: # Alloc empty buffer
      self._buf = self.allocator.alloc(self.dtype.bytes * self.size)
    else:
      self._buf = with_data 
    return self
  def pointer(self, to_type=ctypes.c_byte):
    return ctypes.cast(ctypes.addressof(to_type.from_buffer(self._buf)), ctypes.POINTER(to_type*self.size)).contents
  def copyin(self, src: memoryview): 
    assert self.allocator is not None, f"device must be an allocator, {self.device.name} is a HostDevice"
    self.allocator.copyin(self.pointer(), src)
  def copyout(self, dst: memoryview):
    assert self.allocator is not None, f"device must be an allocator, {self.device.name} is a HostDevice"
    self.allocator.copyout(dst, self.pointer())
  def view(self):
    return Buffer(self.device, self.size, self.dtype) 
  def __repr__(self):
    return f'<real buffer nbytes={self.nbytes} dtype={self.dtype} allocated={self.allocated}> ref_count={self._ref_count}'
  def __hash__(self): return id(self)


@dataclass(frozen=True, eq=True)
class MemBuffer:
  buff: Buffer
  vt: ViewTracker 

@dataclass(frozen=True, eq=True)
class ConstBuffer:
  value: ConstType
  device: Device
  vt: ViewTracker