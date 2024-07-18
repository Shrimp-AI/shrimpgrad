from __future__ import annotations
from collections import defaultdict
import ctypes
from typing import Callable, Generic, Iterable, List, TypeVar, Union, cast
from shrimpgrad import Tensor
from shrimpgrad.device import Buffer, Device, Jitable, MemBuffer
from shrimpgrad.engine.runner import CompiledKernel, shrimp_jit
from shrimpgrad.util import SupportsGetItem

def jit_capture_kernels(kernels: List[CompiledKernel], input_buffers: List[Buffer], device: Union[Device,Jitable]):
  assert isinstance(device, Jitable), f"device {device} is not jittable"
  assert kernels, "no kernels to capture"
  print(f"[JIT_CAPTURE_KERNELS] capturing {len(kernels)} kernels for {device} with {len(input_buffers)} changing inputs")
  native_fxn = device.jitify(kernels, input_buffers)
  print("[DONE JIT_CAPTURE_KERNELS]")
  return native_fxn

def _process_return_type(ret: object):
  if isinstance(ret, Iterable):
    for r in ret: r.realize() 
    return
  assert isinstance(ret, Tensor), 'return type must be a tensor at this point'
  ret.realize()

ReturnType = TypeVar('ReturnType')
class ShrimpJit(Generic[ReturnType]):
  def __init__(self, fn:Callable[..., ReturnType]):
    self.fn = fn
    self.reset()

  def reset(self):
    # The captured CompiledKernel objects
    self.jit_kernels: List[CompiledKernel] = []
    # The number of times this jitted function has
    # executed.
    self.exec_cnt = 0
    self.native_lib = None
    self.replace_buffer = {}

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_tensors: List[Tensor] = [t  for t in args if t.__class__ is Tensor]
    for t in input_tensors: t.realize()
    self.input_buffers = [t.thunk.base.buff for t in input_tensors]
    if self.exec_cnt == 0:
      print("[JIT_IGNORE]")
      # jit ignore
      self.ret = self.fn(*args, **kwargs)
      _process_return_type(self.ret)
    elif self.exec_cnt == 1:
      # jit capture
      print("[JIT_CAPTURE]")
      shrimp_jit.append(self)
      self.ret = self.fn(*args, **kwargs)
      _process_return_type(self.ret)
      shrimp_jit.clear()
      device = self.ret.device if isinstance(self.ret, Tensor) else cast(SupportsGetItem[int, Tensor],self.ret)[0].device
      self.native_fxn = jit_capture_kernels(self.jit_kernels, self.input_buffers, device)
      del self.replace_buffer
    else:
      #  jit exec
      print("[JIT_EXEC]")
      assert self.native_fxn is not None, 'Native function failed to compile!'
      self.native_fxn(*[ctypes.byref(b._pointer(ctypes.c_float)) for b in self.input_buffers])
    self.exec_cnt += 1
    return self.ret

  def jit_capture(self, ck: CompiledKernel):
    b2n = {}
    buffs = defaultdict(list)
    for buff in ck.buffs['input']:
      name = ck.buff2name[buff]
      new_buff = self.add_buffer(buff)
      b2n[new_buff] = name
      buffs['input'].append(new_buff)
    for buff in ck.buffs['output']:
      name = ck.buff2name[buff]
      new_buff = self.add_buffer(buff)
      b2n[new_buff] = name
      buffs['output'].append(new_buff)
    self.jit_kernels.append(CompiledKernel(ck.ir, ck.dev, b2n, buffs, ck.name, ck.batched))

  def add_buffer(self, b: MemBuffer) -> MemBuffer:
    if found:=self.replace_buffer.get(b, None): return found
    if b.__class__ is MemBuffer:
      if b.buff.allocated: return b
      self.replace_buffer[b] = ret = MemBuffer(Buffer(b.buff.device, b.buff.size, b.buff.dtype), b.vt)
    else: return b
    return ret

