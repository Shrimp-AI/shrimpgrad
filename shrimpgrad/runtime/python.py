from __future__ import annotations
import base64
import math
import operator
import pickle
from shrimpgrad.device import Accelerator, Allocator, Compiler, Runtime
from shrimpgrad.engine.lower import LowIRGraph
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, TernaryOps, Op

python_alu = {
  UnaryOps.LOG2: lambda x: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan,
  UnaryOps.EXP2: lambda x: math.exp(x*math.log(2)),
  UnaryOps.SQRT: lambda x: math.sqrt(x) if x >= 0 else math.nan, UnaryOps.SIN: math.sin,
  UnaryOps.NEG: lambda x: (not x) if isinstance(x, bool) else -x,
  BinaryOps.MUL: operator.mul, BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.XOR: operator.xor,
  BinaryOps.MAX: max, BinaryOps.CMPEQ: operator.eq, BinaryOps.CMPLT: operator.lt,
  BinaryOps.MOD: lambda x,y: abs(int(x))%abs(int(y))*(1,-1)[x<0],
  BinaryOps.DIV: lambda x,y: int(x/y) if isinstance(x, int) else (x/y if y != 0 else x*math.inf),
  TernaryOps.WHERE: lambda x,y,z: y if x else z}

class PythonDevice(Accelerator):
  def __init__(self):
    super().__init__("PYTHON", PythonAllocator, PythonRenderer, PythonCompiler, PythonRuntime)
  def compiler(self) -> PythonCompiler: return self._compiler()
  def allocator(self) -> PythonAllocator: return self._allocator()
  def runtime(self) -> PythonRuntime: return self._runtime()
  def renderer(self) -> PythonRenderer: return self._renderer()

class PythonRenderer:
  def render(self, ir_graph: LowIRGraph) -> str: return base64.b64encode(pickle.dumps(ir_graph)).decode()

class PythonCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonRuntime(Runtime):
  def exec(self, lib: bytes):
    ir_graph = pickle.loads(lib)
    return


class PythonAllocator(Allocator):
  def alloc(self, size:int): return memoryview(bytearray(size))
  def copyin(self, dst, src: memoryview): dst[:] = src[:]
  def copyout(self, dst:memoryview, src): dst[:] = src[:]
  def free(self): return