from __future__ import annotations
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from shrimpgrad.device import Device, MemBuffer
from shrimpgrad.engine.lower import LowIRGraph, LowerFusedKernel
from shrimpgrad.engine.scheduler import FusedKernel, FusedKernelBuilder
from shrimpgrad.future import Thunk
from shrimpgrad.runtime.ops import LoadOps

buff_to_name: Optional[Dict[Any, str]
] = None

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

def realize(out: Thunk):
  kernels: List[CompiledKernel] = []
  sched = _schedule(out)
  load_kerns, unkerned = _gen_load_kernels(sched)
  for load_kern in load_kerns:
    load_kern()
  buffers: List[DefaultDict[str, List[MemBuffer]]] = []
  func_names = set() 
  names = []
  for i, s in enumerate(unkerned):
    buffs = defaultdict(list)
    for inp in s.computation.ins:
      for buf in inp:
        buffs['input'].append(buf)
    for obuf in s.computation.out:
      buffs['output'].append(obuf)
    buffers.append(buffs)
    name = [op.name.lower() for op in s.computation.ops] 
    in_shape = s.computation.ins[0][0].vt.shape 
    out_shape = s.computation.out[-1].vt.shape
    name.append('_'.join([str(d) for d in (in_shape if in_shape else [0])]))
    name.append('_'.join([str(d) for d in (out_shape if out_shape else [0])]))
    func_name = '_'.join(name + [str(i)])
    assert func_name not in func_names, f'naming collision for {func_name}' 
    func_names.add(func_name)
    names.append(func_name)
  
  ir_graphs = _lower(unkerned)
  # Delay compilation to batch compile the kernels
  batched = True
  for irg, buffs, name in zip(ir_graphs, buffers, func_names):
    kernels.append(CompiledKernel(irg, out.device, buff_to_name, buffs, name=name, batched=batched))

  if batched and kernels:
    for kernel in kernels:
      if shrimp_jit: shrimp_jit[0].jit_capture(kernel)
    batched_kernel = BatchedCompiledKernel(kernels)
    batched_kernel()
    return

  for kernel in kernels:
    if shrimp_jit: shrimp_jit[0].jit_capture(kernel)
    kernel()

def _gen_load_kernels(schedule: List[FusedKernel]) ->  Tuple[List[BufferCopy], List[FusedKernel]]:
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

class BatchedCompiledKernel:
  def __init__(self, cks: List[CompiledKernel]):
    self.dev = cks[0].dev 
    self.buff2name = cks[0].buff2name 
    self.buffs = [ck.buffs for ck in cks] 
    self.names = [ck.name for ck in cks] 
    self.prgs = [ck.prg for ck in cks]
    self.src = '\n'.join([ck.prg.src for ck in cks]) 
    try:
      self.lib = self.dev.compiler().cached_compile(self.src)
    except Exception as e:
      print(self.src)
      raise e
  def __repr__(self) -> str: return f"<BatchedCompiledKernel id={id(self)}>"
  def __call__(self):
    try:
      self.dev.runtime().batched_exec(self.lib, self.names, self.buffs, self.buff2name, [prg.args2pos for prg in self.prgs])
    except Exception as e:
      print(self.src)
      raise e

class CompiledKernel:
  def __init__(self, ir: LowIRGraph, dev: Device, buff_to_name, buffs: DefaultDict[str, List[MemBuffer]], name=None, batched=False) -> None:
    self.ir, self.dev, self.buffs = ir, dev, buffs
    self.name = name 
    self.prg = self.dev.renderer().render(self.ir, self.name)
    self.batched = batched
    if not self.batched:
      # Delay compilation
      try:
        self.lib = self.dev.compiler().cached_compile(self.prg.src)
      except Exception as e:
        print(self.prg.src)
        raise e
    self.buff2name = buff_to_name
  def __repr__(self) -> str: return f"<CompiledKernel id={id(self)}>"
  def __call__(self):
    try:
      self.rt = self.dev.runtime().exec(self.lib, self.prg.fname if self.name is None else self.name, self.buffs, self.buff2name, self.prg.args2pos)
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