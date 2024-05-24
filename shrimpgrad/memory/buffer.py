import ctypes

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
  @property
  def allocated(self): return hasattr(self, '_buf')
  def allocate(self, with_data=None):
    if with_data is None: # Alloc empty buffer
      self._buf = self.allocator.alloc(self.dtype.bytes * self.size)
    else:
      self._buf = with_data 
    return self
  def pointer(self, to_type=ctypes.c_byte):
    return ctypes.cast(ctypes.addressof(to_type.from_buffer(self._buf)), ctypes.POINTER(to_type*self.size)).contents
  def copyin(self, src: memoryview): 
    self.allocator.copyin(self.pointer(), src)
  def copyout(self, dst: memoryview):
    self.allocator.copyout(dst, self.pointer())
  def view(self):
    return Buffer(self.allocator, self.size, self.dtype) 
  def __repr__(self):
    return f'<real buffer nbytes={self.dtype.bytes*self.size} dtype={self.dtype} allocated={self.allocated}> ref_count={self._ref_count}'
  def __hash__(self): return id(self)