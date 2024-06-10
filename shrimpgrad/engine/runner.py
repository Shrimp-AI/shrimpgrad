from shrimpgrad.device import Accelerator, Buffer
from shrimpgrad.engine.lower import LowerFusedKernel
from shrimpgrad.engine.scheduler import FusedKernelBuilder, print_schedule
from shrimpgrad.future import Thunk

class Runner:
  def __init__(self, out: Thunk, debug=False) -> None:
    assert isinstance(out.device, Accelerator), f'runner requires an accelerator given: {out.device.name}'
    fkb = FusedKernelBuilder(out)
    self.schedule = fkb.schedule()
    if debug: print_schedule(self.schedule)
    fkb = FusedKernelBuilder(out)
    schedule = fkb.schedule()
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    for i, ir_graph in enumerate(ir_graphs):
      if debug:
        print(f"GRAPH {i}")
      ir_graph.print()
      src = out.device.renderer().render(ir_graph)
      lib = out.device.compiler().compile(src)
      out.device.runtime().exec(lib)


class BufferCopy:
  def __init__(self, dst: Buffer, src: Buffer, size: int):
    assert dst.size == src.size and dst.dtype == src.dtype, f"buffer copy mismatch, {dst.size} != {src.size}, {dst.dtype} != {src.dtype}"
    self.dst, self.src, self.size = dst, src, size
  def copy(self):
    assert self.src.allocated, 'src buffer needs to be allocated'
    if not self.dst.allocated:
      self.dst.allocate()
    self.dst.copyin(self.src.as_buffer())

  def __call__(self):
    self.copy()