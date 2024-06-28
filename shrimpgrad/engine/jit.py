
from typing import Callable, Generic, List, TypeVar

from shrimpgrad import Tensor
from shrimpgrad.device import Jitable
from shrimpgrad.engine.runner import CompiledKernel, shrimp_jit




class JitExec:
  pass


def jit_capture_kernels(kernels: List[CompiledKernel], device: Jitable):
  assert isinstance(device, Jitable), f"device {device} is not jittable"
  assert kernels, "no kernels to capture"
  print(f"[JIT_CAPTURE_KERNELS] capturing {len(kernels)} kernels for {device}")
  device.jitify(kernels)
  print("[DONE JIT_CAPTURE_KERNELS]")

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

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_tensors = [(name, t) for name, t in enumerate(args) if t.__class__ is Tensor]
    for _, t in input_tensors: t.realize()
    if self.exec_cnt == 0:
      print("[JIT_IGNORE]")
      # jit ignore
      self.ret = self.fn(*args, **kwargs)
    elif self.exec_cnt == 1:
      # jit capture
      shrimp_jit.append(self)
      self.ret = self.fn(*args, **kwargs)
      shrimp_jit.clear()
      device = self.ret.device
      jit_capture_kernels(self.jit_kernels, device)
    else:
      #  jit exec
      print("[JIT_EXEC]")

    self.exec_cnt += 1
    return self.ret

  def jit_capture(self, ck: CompiledKernel):
    self.jit_kernels.append(CompiledKernel(ck.ir, ck.dev, ck.buff2name, ck.buffs))
