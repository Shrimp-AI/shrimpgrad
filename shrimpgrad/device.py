
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from typing import Type
from shrimpgrad.dtype import ConstType, DType, dtypes
from shrimpgrad.meta.singleton import Singleton
from shrimpgrad.view import ViewTracker
import array

class Device(metaclass=Singleton):
  def __init__(self, name:str): self.name = name
  def __eq__(self, other: Device): return self.name == other.name
  def __hash__(self): return id(self)

class Allocator:
  def alloc(self): raise NotImplementedError('implement alloc')
  def free(self): raise NotImplementedError('implement free')
  def copyin(self): raise NotImplementedError('implement copyin')
  def copyout(self): raise NotImplementedError('implement copyout')

class Renderer:
  def render(self): raise NotImplementedError('implement render')

class Compiler:
  def compile(self): raise NotImplementedError('implement compile')

class Runtime:
  def exec(self): raise NotImplementedError('implement exec')

class Accelerator(Device):
  def __init__(self, name:str, allocator: Type[Allocator], renderer: Type[Renderer], compiler: Type[Compiler], runtime: Type[Runtime]) -> None:
    super().__init__(name)
    self._allocator, self._compiler, self._runtime, self._renderer = allocator, compiler, runtime, renderer
  def compiler(self) -> Compiler: raise NotImplementedError('implement compiler for accelerator')
  def allocator(self) -> Allocator: raise NotImplementedError('implement allocator for accelerator')
  def runtime(self) -> Runtime: raise NotImplementedError('implement runtime for accelerator')
  def renderer(self): raise NotImplementedError('implement renderer for accelerator')

class HostDevice(Device):
  def copyto(self, accelerator: Accelerator): raise NotImplementedError('implement copyto')

class CPU(HostDevice):
  def __init__(self):
    super().__init__("CPU")
    self._allocator = MallocAllocator
  def copyto(self, accelerator: Accelerator): pass
  def allocator(self): return self._allocator()

class MallocAllocator(Allocator):
  def alloc(self, size:int):
    return (ctypes.c_uint8 * size)()
  def copyin(self, dst, src:memoryview):
    ctypes.memmove(dst, from_mv(src), len(src))
  def copyout(self, dst:memoryview, src):
    ctypes.memmove(from_mv(dst), src, len(dst))
  def free(self): return

class Buffer:
  def __init__(self, device: Device, size:int, dtype: DType):
    self.device = device
    self.allocator, self.dtype, self.size = device.allocator(), dtype, size
    self._ref_count = 1
  @property
  def allocated(self): return hasattr(self, '_buf')
  @property
  def nbytes(self): return self.size * self.dtype.bytes
  def allocate(self, with_data=None):
    if with_data is None: # Alloc empty buffer
      self._buf = self.allocator.alloc(self.dtype.bytes * self.size)
    else:
      self._buf = memoryview(arr:=array.array('d', with_data))
      print(f"{arr = }")
      self._buf = self._buf.cast("B", shape=(self._buf.nbytes,))
    return self
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, 'as_buffer'): return self.allocator.as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def pointer(self, to_type=ctypes.c_byte):
    if not isinstance(self._buf, memoryview): self._buf = memoryview(self._buf)
    return ctypes.cast(ctypes.addressof(to_type.from_buffer(self._buf)), ctypes.POINTER(to_type*self.size)).contents
  def copyin(self, src: memoryview):
    assert self.allocator is not None, f"device must be an allocator, {self.device.name} is a HostDevice"
    src = src.cast("B", shape=(src.nbytes,))
    self.allocator.copyin(self._buf, src)
    return self._buf
  def copyout(self, dst: memoryview):
    assert self.allocator is not None, f"device must be an allocator, {self.device.name} is a HostDevice"
    dst = dst.cast("B", shape=(dst.nbytes,))
    self.allocator.copyout(dst, from_mv(self._buf))
    return dst
  def view(self):
    return Buffer(self.device, self.size, self.dtype)
  def __repr__(self):
    return f'<real buffer nbytes={self.nbytes} dtype={self.dtype} allocated={self.allocated}> ref_count={self._ref_count}'
  def __hash__(self): return id(self)

def from_mv(mv:memoryview, to_type=ctypes.c_char):
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
@dataclass(frozen=True, eq=True)
class MemBuffer:
  buff: Buffer
  vt: ViewTracker

@dataclass(frozen=True, eq=True)
class ConstBuffer:
  value: ConstType
  device: Device
  vt: ViewTracker

