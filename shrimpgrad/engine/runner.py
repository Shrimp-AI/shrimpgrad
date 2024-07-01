from __future__ import annotations
from collections import defaultdict
from typing import DefaultDict, List
from shrimpgrad.device import Accelerator, ConstBuffer, MemBuffer
from shrimpgrad.engine.lower import LowIRGraph, LowerFusedKernel
from shrimpgrad.engine.scheduler import FusedKernel, FusedKernelBuilder
from shrimpgrad.future import Thunk
from shrimpgrad.runtime.ops import LoadOps

buff_to_name = None

def buff2name(buff) -> str:
  if buff in buff_to_name:
    return buff_to_name[buff]
  return str(buff)

def _schedule(out: Thunk) -> List[FusedKernel]:
  return FusedKernelBuilder(out).schedule()

def _lower(schedule: List[FusedKernel]) -> List[LowIRGraph]:
  global buff_to_name
  irgs = (lkb:=LowerFusedKernel(schedule)).lower()
  buff_to_name = lkb.node_to_symbol
  return irgs

# When running with JIT, the JIT engine will
# add itself to this list, allowing the runner
# to add the kernel to the JIT engine.
shrimp_jit = []

def realize(out: Thunk):
  kernels: List[CompiledKernel|BufferCopy] = []
  sched = _schedule(out)
  load_kerns, unkerned = _gen_load_kernels(sched)
  for load_kern in load_kerns:
    load_kern()
  buffers: List[DefaultDict[str, List[MemBuffer|ConstBuffer]]] = []
  for s in unkerned:
    buffs = defaultdict(list)
    for inp in s.computation.ins:
      for buf in inp:
        buffs['input'].append(buf)
    for obuf in s.computation.out:
      buffs['output'].append(obuf)
    buffers.append(buffs)
  ir_graphs = _lower(unkerned)
  for irg, buffs in zip(ir_graphs, buffers):
    kernels.append(CompiledKernel(irg, out.device, buff_to_name, buffs))
  for kernel in kernels:
    if shrimp_jit: shrimp_jit[0].jit_capture(kernel)
    kernel()

def _gen_load_kernels(schedule: List[FusedKernel]) -> None:
  load_kernels = []
  un_kerneled = []
  for fk in schedule:
    if fk.computation.ops[0] == LoadOps.CONST: continue
    if len(fk.computation.ins) == 1 and fk.computation.ops[0] == LoadOps.COPY:
      src = fk.computation.ins[0][0]
      dst = fk.computation.out[0]
      size = fk.computation.args[0]

      assert isinstance(src, MemBuffer)
      assert isinstance(dst, MemBuffer)
      load_kernels.append(BufferCopy(dst, src, size))
    else: un_kerneled.append(fk)
    # Allocate output buffers
    for buff in fk.computation.out:
      if isinstance(buff, MemBuffer):
        if not buff.buff.allocated:
          buff.buff.allocate()
  return load_kernels, un_kerneled

class CompiledKernel:
  def __init__(self, ir: LowIRGraph, dev: Accelerator, buff_to_name, buffs: DefaultDict[str, List[MemBuffer | ConstBuffer]]
) -> None:
    self.ir, self.dev, self.buffs = ir, dev, buffs
    self.prg = self.dev.renderer().render(self.ir)
    self.lib = self.dev.compiler().cached_compile(self.prg.src)
    self.buff2name = buff_to_name
  def __repr__(self) -> str: return f"<CompiledKernel id={id(self)}>"
  def __call__(self):
    try:
      self.rt = self.dev.runtime().exec(self.lib, self.prg.fname, self.buffs, self.buff2name, self.prg.args2pos)
    except Exception as e:
      print(self.prg.src)
      raise e

class BufferCopy:
  def __init__(self, dst: MemBuffer, src: MemBuffer, size: int):
    assert dst.buff.size == src.buff.size and dst.buff.dtype == src.buff.dtype, f"buffer copy mismatch, {dst.buff.size} != {src.buff.size}, {dst.buff.dtype} != {src.buff.dtype}"
    self.dst, self.src, self.size = dst, src, size
  def copy(self):
    assert self.src.buff.allocated, 'src buffer needs to be allocated'
    if not self.dst.buff.allocated: self.dst.buff.allocate()
    self.dst.buff.copyin(self.src.buff.as_buffer())
  def __str__(self) -> str:
    return f"<BufferCopyKernel dst={buff2name(self.dst)} src={buff2name(self.src)} nbytes={self.size}>"
  def __call__(self): self.copy()