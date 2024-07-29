from __future__ import annotations
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
from shrimpgrad.device import MemBuffer
from shrimpgrad.engine.lower import LowIRGraph, LowerFusedKernel
from shrimpgrad.engine.scheduler import FusedKernel, FusedKernelBuilder, print_schedule
from shrimpgrad.future import Thunk
from shrimpgrad.runtime.ops import LoadOps
from shrimpgrad.knobs import DEBUG

buff_to_name: Optional[Dict[Any, str]] = None

def buff2name(buff) -> str:
  assert buff_to_name is not None, "buff_to_name must be allocated"
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

def realize(out: Thunk, batched=True):
  sched = _schedule(out)
  if DEBUG >= 4: print_schedule(sched)
  buff_copies, unkerned = _gen_load_kernels(sched)
  [buffcpy() for buffcpy in buff_copies]
  buffers = map_buffers_to_kernel(unkerned)
  func_names = name_kernels(unkerned)
  ir_graphs = _lower(unkerned)
  if DEBUG >= 4: 
    for irg in ir_graphs: irg.print()
  k = [CompiledKernel(irg, out.device, buff_to_name, buffs, name=name, 
    batched=batched) for irg, buffs, name in zip(ir_graphs, buffers, 
    func_names)]
  if batched and k:
    [shrimp_jit[0].jit_capture(kernel) for kernel in k if shrimp_jit]
    BatchedCompiledKernel(k)()
    return
  [shrimp_jit[0].jit_capture(kernel) for kernel in k if shrimp_jit]
  [kernel() for kernel in k]

def map_buffers_to_kernel(kernels: List[FusedKernel]) -> List[DefaultDict[str, List[MemBuffer]]]:
  return [defaultdict(list, {
    'input': [buf for inp in s.computation.ins for buf in inp],
    'output': [obuf for obuf in s.computation.out]
  }) for s in kernels]

def name_kernels(kernels: List[FusedKernel]) -> List[str]:
  func_names: Set[str] = set()
  return [
    func_name for func_name in (
      '_'.join(
        [op.name.lower() for op in s.computation.ops] +
        ['0' if s.computation.ops[0] in [LoadOps.CONST,LoadOps.CONTIGUOUS] else '_'.join(
          map(str, s.computation.ins[0][0].vt.shape))] +
        ['_'.join(map(str, s.computation.out[-1].vt.shape))] +
        [str(i)]
      ) for i, s in enumerate(kernels)
    ) if func_name not in func_names and not func_names.add(func_name)]

def _gen_load_kernels(schedule: List[FusedKernel]) -> Tuple[List[BufferCopy], List[FusedKernel]]:
  l, u = [], []
  for fk in schedule:
    c = fk.computation
    if c.ops[0] == LoadOps.CONST and c.out[0].buff.allocated: continue
    if len(c.ins) == 1 and c.ops[0] == LoadOps.COPY:
      l.append(BufferCopy(c.out[0], c.ins[0][0], c.args[0]))
    else: u.append(fk)
    [buff.buff.allocate() for buff in c.out if not buff.buff.allocated]
  return l, u

class BaseKernel:
  def __init__(self, dev, buffs, name=None):
    self.dev, self.buffs, self.name = dev, buffs, name
  def __repr__(self) -> str: return f"<{self.__class__.__name__} id={id(self)}>"
  def compile(self, src):
    if DEBUG >= 4: print(src)
    try: return self.dev.compiler().cached_compile(src)
    except Exception as e:
      print(src)
      raise e

class CompiledKernel(BaseKernel):
  def __init__(self, ir, dev, buff_to_name, buffs, name=None, batched=False):
    super().__init__(dev, buffs, name)
    self.ir, self.prg, self.batched = ir, dev.renderer().render(ir, name), batched
    self.buff2name = buff_to_name
    if not batched: self.lib = self.compile(self.prg.src)
  def __call__(self):
    try: self.rt = self.dev.runtime().exec(self.lib, self.prg.fname if self.name is None else self.name, self.buffs, self.buff2name, self.prg.args2pos)
    except Exception as e: 
      print(self.prg.src)
      raise e

class BatchedCompiledKernel(BaseKernel):
  def __init__(self, cks: List[CompiledKernel]):
    super().__init__(cks[0].dev, [ck.buffs for ck in cks])
    self.buff2name, self.names, self.prgs = cks[0].buff2name, [ck.name for ck in cks], [ck.prg for ck in cks]
    self.src, self.lib = '\n'.join([ck.prg.src for ck in cks]), self.compile('\n'.join([ck.prg.src for ck in cks]))
  def __call__(self):
    try: self.dev.runtime().batched_exec(self.lib, self.names, self.buffs, self.buff2name, [prg.args2pos for prg in self.prgs])
    except Exception as e: 
      print(self.src)
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