
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from typing import Optional, Type
from shrimpgrad.dtype import ConstType, DType
from shrimpgrad.meta.singleton import Singleton
from shrimpgrad.view import ViewTracker
import numpy as np
from numpy.typing import NDArray
from collections.abc import Buffer as BufferType

"""
 Base Classes for Device

 A host device is the CPU on the executing machine, mainly used for copying
 user provided data into a buffer.

 An acclerator device is any stack of compiler, allocator, runtime, renderer
 and jittable with the purpose of speeding up inference and training.

 For example, the ClangDevice provides support for executing Shrimp Tensor computations
 via C.

 Devices are singletons.
"""
class Device(metaclass=Singleton):
  def __init__(self, name:str, allocator: Type[Allocator], renderer: Type[Renderer], compiler: Type[Compiler], runtime: Type[Runtime]) -> None:
    self.name = name
    self._allocator, self._compiler, self._runtime, self._renderer = allocator, compiler, runtime, renderer

  def compiler(self) -> Compiler: raise NotImplementedError('implement compiler for accelerator')
  def allocator(self) -> Allocator: raise NotImplementedError('implement allocator for accelerator')
  def runtime(self) -> Runtime: raise NotImplementedError('implement runtime for accelerator')
  def renderer(self) -> Renderer: raise NotImplementedError('implement renderer for accelerator')

  def __eq__(self, other: Device): return self.name == other.name
  def __hash__(self): return id(self)
  def __repr__(self): return self.name

class Allocator:
  def alloc(self, size:int) -> BufferType: raise NotImplementedError('implement alloc')
  def free(self): raise NotImplementedError('implement free')
  def copyin(self, dst, src): raise NotImplementedError('implement copyin')
  def copyout(self, dst, src): raise NotImplementedError('implement copyout')

class Renderer:
  def render(self, ir, name): raise NotImplementedError('implement render')

class Compiler(metaclass=Singleton):
  def __init__(self):
    self.cache = {}
  def compile(self, src: str): raise NotImplementedError('implement compile')
  def cached_compile(self, src: str, **kwargs):
    if src in self.cache: return self.cache[src]
    self.cache[src] = self.compile(src, **kwargs)
    return self.cache[src]

class Runtime(metaclass=Singleton):
  def exec(self, *args, **kwargs): raise NotImplementedError('implement exec')
  def batched_exec(self, *args, **kwargs): raise NotImplementedError('implement batched_exec')

# A mixin for devices that want the capability of executing native only.
class Jitable:
  def jitify(self): raise NotImplementedError('implement jitify')

class CPU(Device):
  def __init__(self):
    super().__init__("CPU", CPUAllocator, Renderer, Compiler, Runtime)
  def allocator(self): return self._allocator()
  def compiler(self): return self.compiler()
  def runtime(self): return self._runtime()
  def renderer(self): return self._renderer()

class CPUAllocator(Allocator):
  def alloc(self): return
  def free(self): return
  def copyin(self): return
  def copyout(self, dst: memoryview, src: NDArray):
    x = np.require(src, requirements='C').data
    x = x.cast("B", shape=(x.nbytes,))
    dst[:] = x

class MallocAllocator(Allocator):
  def alloc(self, size:int) -> ctypes.Array:
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
    self._buf: Optional[BufferType] = None
    self._ref_count = 1
  @property
  def allocated(self): return self._buf is not None 

  @property
  def nbytes(self): return self.size * self.dtype.bytes

  def allocate(self, with_data=None):
    if with_data is None: # Alloc empty buffer
      self._buf = self.allocator.alloc(self.dtype.bytes * self.size)
    else:
      self._buf = np.array(with_data, np.float32)
    return self

  def as_buffer(self ) -> memoryview: return self.copyout(memoryview(bytearray(self.nbytes)))

  def pointer(self, to_type:Type=ctypes.c_char):
    assert self._buf is not None, "buffer is not allocated" 
    if not isinstance(self._buf, memoryview): self._buf = memoryview(self._buf)
    return self._buf

  def _pointer(self, to_type):
    assert self._buf is not None, "buffer is not allocated"
    if not isinstance(self._buf, memoryview): self._buf = memoryview(self._buf)
    return ctypes.cast(ctypes.addressof(to_type.from_buffer(self._buf)), ctypes.POINTER(to_type*self.size)).contents

  def copyin(self, src: memoryview):
    assert self.allocator is not None, f"device must be an allocator, {self.device.name} is a HostDevice"
    assert self._buf is not None, "buffer is not allocated"
    src = src.cast("B", shape=(src.nbytes,))
    self.allocator.copyin(self._buf, src)
    return self._buf

  def copyout(self, dst: memoryview):
    assert self.allocator is not None, f"device must be an allocator, {self.device.name} is a HostDevice"
    dst = dst.cast("B", shape=(dst.nbytes,))
    self.allocator.copyout(dst, self._buf)
    return dst

  def view(self): return Buffer(self.device, self.size, self.dtype)

  def __repr__(self):
    return f'<real buffer nbytes={self.nbytes} dtype={self.dtype} allocated={self.allocated}> ref_count={self._ref_count}'

def from_mv(mv:memoryview, to_type=ctypes.c_char):
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents

@dataclass(frozen=True, eq=True)
class MemBuffer:
  buff: Buffer
  vt: ViewTracker

@dataclass(frozen=False)
class ConstBuffer:
  value: ConstType
  device: Device
  vt: ViewTracker
  def __hash__(self): return hash(id(self))

