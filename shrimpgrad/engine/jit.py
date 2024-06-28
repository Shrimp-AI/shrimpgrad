from __future__ import annotations
import ctypes
from typing import Callable, Generic, List, TypeVar
from shrimpgrad import Tensor
from shrimpgrad.device import Buffer, Jitable
from shrimpgrad.engine.runner import CompiledKernel, shrimp_jit

def jit_capture_kernels(kernels: List[CompiledKernel], input_buffers: List[Buffer], device: Jitable):
  assert isinstance(device, Jitable), f"device {device} is not jittable"
  assert kernels, "no kernels to capture"
  print(f"[JIT_CAPTURE_KERNELS] capturing {len(kernels)} kernels for {device} with {len(input_buffers)} changing inputs")
  native_fxn = device.jitify(kernels, input_buffers)
  print("[DONE JIT_CAPTURE_KERNELS]")
  return native_fxn

def _process_return_type(ret: ReturnType):
  if ret.__class__ in [List, tuple]: 
    for r in ret: r.realize() if r.thunk.base.realized is None else '' 
  elif ret.thunk.realized is None: ret.realize()
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

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_tensors: List[Tensor] = [(name, t) for name, t in enumerate(args) if t.__class__ is Tensor]
    input_buffers = [t.thunk.base.buff if not t.is_scalar() else t.thunk.base.cbuff for _, t in input_tensors]
    for _, t in input_tensors: t.realize()
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
      device = self.ret.device if self.ret.__class__ is Tensor else self.ret[0].device
      self.native_fxn = jit_capture_kernels(self.jit_kernels, input_buffers, device)
    else:
      #  jit exec
      print("[JIT_EXEC]")
      assert self.native_fxn is not None, 'Native function failed to compile!'
      self.native_fxn(*[b._pointer(ctypes.c_float) for b in input_buffers])
    self.exec_cnt += 1
    return self.ret

  def jit_capture(self, ck: CompiledKernel):
    self.jit_kernels.append(CompiledKernel(ck.ir, ck.dev, ck.buff2name, ck.buffs))

