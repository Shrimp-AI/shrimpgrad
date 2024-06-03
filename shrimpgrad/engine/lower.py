from __future__ import annotations
from ctypes import Union
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Sequence, Tuple
from shrimpgrad.device import ConstBuffer, MemBuffer
from shrimpgrad.dtype import ConstType, DType, dtypes
from shrimpgrad.engine.scheduler import FusedKernel
from shrimpgrad.runtime.ops import BinaryOps, LoadOps, ReduceOps, TernaryOps, UnaryOps

class LowIR(Enum):
  GLOBAL = auto()
  ACC = auto()
  ADDRESS = auto()
  LOCAL = auto()
  LOAD = auto()
  CONST = auto() 
  STORE = auto()
  ALU = auto()
  BEGIN_LOOP = auto() 
  END_LOOP = auto() 
  IF = auto() 
  ENDIF = auto() 

@dataclass(frozen=True, eq=True)
class Node:
  op: LowIR
  ancestors: Tuple[Node,...]

@dataclass(frozen=True, eq=True)
class ConstNode(Node):
  dtype: DType
  val: ConstType

@dataclass(frozen=True, eq=True)
class GlobalNode(Node):
  name: str
  dtype: DType
  pos: int
  mutable: bool

@dataclass(frozen=True, eq=True)
class LocalNode(Node):
  name: str
  dtype: DType

@dataclass(frozen=True, eq=True)
class AddressNode(Node):
  idx: int
  stride: int
  step: int

@dataclass(frozen=True, eq=True)
class LoadNode(Node):
  pass

@dataclass(frozen=True, eq=True)
class AccumulatorNode(Node):
  name: str
  dtype: DType
  alu: BinaryOps

@dataclass(frozen=True, eq=True)
class StoreNode(Node):
  pass

@dataclass(frozen=True, eq=True)
class ALUNode(Node):
  alu: Union[BinaryOps, UnaryOps, TernaryOps]
  dtype: DType

@dataclass(frozen=True, eq=True)
class BeginLoopNode(Node):
  pass

@dataclass(frozen=True, eq=True)
class EndLoopNode(Node):
  pass


# A graph where each node occupies an index (based on the order of addition)
# and has 0-to-Many back pointers to dependecies via node.ancestors
class LowIRGraph:
  def __init__(self):
    self.G: List[Node] = [] 
  
  def const(self, dtype: DType, val: ConstType) -> Node:  
    self.G.append(node:=ConstNode(LowIR.CONST, (), dtype, val))
    return node

  def define_global(self, name:str, dtype: DType, mutable: bool, pos: int) -> Node:
    self.G.append(node:=GlobalNode(LowIR.GLOBAL, (), name, dtype, pos, mutable))
    return node

  def local_var(self, name: str, dtype: DType, val: ConstNode) -> Node:
    self.G.append(node:=LocalNode(LowIR.LOCAL, (val, ), name, dtype))
    return node

  def address(self, idxs: List[LocalNode], strides: Tuple[int,...], step: int):
    self.G.append(node:=AddressNode(LowIR.ADDRESS, (), idxs, strides, step))
    return node

  def load(self, node: Union[GlobalNode, LocalNode], address: AddressNode) -> Node:
    self.G.append(node:=LoadNode(LowIR.LOAD, (node, address))) 
    return node

  def accumulator(self, alu: BinaryOps, 
                  name: str, dtype: DType, 
                  operands: Sequence[Union[ConstNode, LoadNode]]) -> Node: 
    self.G.append(node:=AccumulatorNode(LowIR.ACC, tuple(operands), name, dtype, alu))
    return node

  def store(self, lhs: Union[GlobalNode, LocalNode],
            address: AddressNode, 
            rhs: Union[LoadNode, ConstNode, LocalNode]) -> Node:
    self.G.append(node:=StoreNode(LowIR.STORE, (lhs, address, rhs)))
    return node

  def alu(self, alu: Union[BinaryOps, UnaryOps, TernaryOps], dtype: DType, *operands) -> Node:
    self.G.append(node:=ALUNode(LowIR.ALU, tuple(operands), alu, type))
    return node

  def begin_loop(self, start: LocalNode, end: ConstNode):
    self.G.append(node:=BeginLoopNode(LowIR.BEGIN_LOOP, (start, end)))
    return node

  def end_loop(self, loop: BeginLoopNode):
    self.G.append(node:=EndLoopNode(LowIR.END_LOOP, (loop, )))
    return node

class LowerFusedKernel:
  def __init__(self, fused_kernels: List[FusedKernel]):
    self.fused_kernels = fused_kernels
    self.symbol_table: Dict[str, Node] = {}
    self.node_to_symbol: Dict[Any, str] = {}
    self.global_counter = 0
    self.local_counter = 0
    self.accum_counter = 0
    self.g = LowIRGraph()
    # TODO: Get dtype from inputs or outputs
    self.dtype = dtypes.float32
    self.consts = []
   
  def global_name(self):
    name = f"glob{self.global_counter}"
    self.global_counter += 1
    return name
  
  def local_name(self):
    name = f"loc{self.local_counter}"
    self.local_counter += 1
    return name
 
  def accum_name(self):
    name = f"acc{self.accum_counter}"
    self.accum_counter += 1
    return name
  
  def lower_inputs(self, inputs: Sequence[MemBuffer, ConstBuffer]):  
    lowered = []
    for inp in inputs:
      if isinstance(inp, MemBuffer):
        g0 = self.lower_copy(inp) 
        lowered.append(g0)
      else:
        c0 = self.lower_const(inp)
        self.consts.append(c0)
        lowered.append(c0)
    return lowered

  def lower_const(self, cbuff: ConstBuffer) -> ConstNode:
    c0 = self.g.const(self.dtype, cbuff.val)
    self.consts.append(c0)
    self.node_to_symbol[cbuff] = len(self.consts) - 1 
    return c0 

  def lower_copy(self, mbuff: MemBuffer) -> GlobalNode:
    g0 = self.g.define_global(name:=self.global_name(), self.dtype, False, self.global_counter-1)
    self.symbol_table[name] = g0
    self.node_to_symbol[mbuff] = name
    return g0

  def lower_bop(self,
                in0: Union[ConstBuffer, MemBuffer],
                in1: Union[ConstBuffer, MemBuffer],
                out0: Union[ConstBuffer, MemBuffer],
                idxs: List[LocalNode], 
                strides0: Tuple[int,...],
                strides1: Tuple[int,...],
                strides2: Tuple[int,...],
                step: int,
                alu: BinaryOps):
    # Define a global for the output
    out1 = self.g.define_global(name:=self.global_name(), self.dtype, False, self.global_counter-1)
    self.symbol_table[name] = out1
    self.node_to_symbol[out0] = name

    # Generate indexes for the two inputs
    addr0 = self.g.address(idxs, strides0, step)
    addr1 = self.g.address(idxs, strides1, step) 

    # Generate two loads from the inputs with the indexes
    # Inputs have previously been defined by copy/const/ or outs of other
    # ops
    g0 = self.symbol_table[self.node_to_symbol[in0]]
    g1 = self.symbol_table[self.node_to_symbol[in1]]
    l0 = self.g.load(g0, addr0)
    l1 = self.g.load(g1, addr1)

    # Generate the alu operation on the two loads
    alu0 = self.g.alu(alu, dtypes.float32, l0, l1)

    # Generate the index for the output
    addr2 = self.g.address(idxs, strides2, step)

    # Store the alu output into the global at addr2 
    self.g.store(out1, addr2, alu0)

  def lower_uop(self, uop):
    pass
  def lower_top(self, top):
    pass
  def lower_rop(self, rop): 
    pass
  def lower_store(self, store):
    pass

  def lower_start_loops(self, ndim:int, shape: Tuple[int,...]):
    loops, idxs = [], []
    for dim in range(ndim):
      # Create two constant values for the loop index
      c0 = self.g.const(dtypes.int32, 0) # start
      c1 = self.g.const(dtypes.int32, shape[dim]) # end
      # Create a loop var init with 0
      l0 = self.g.local_var(self.local_name(), dtypes.int32, c0)
      idxs.append(l0)
      # Begin a loop from var = 0 to c1
      loop = self.g.begin_loop(l0, c1)
      loops.append(loop)
    return loops, idxs

  def lower_end_loops(self, loops: List[BeginLoopNode]):
    for loop in loops: self.g.end_loop(loop)

  
  def lower_single_op_kernel(self, fused_kernel: FusedKernel):
    assert len(fused_kernel.computation.ins) == 1
    assert len(fused_kernel.computation.out) == 1
    inputs = fused_kernel.computation.ins[0]
    output = fused_kernel.computation.out[0]
    op = fused_kernel.computation.ops[0]
    arg = fused_kernel.computation.args[0]
    if op is LoadOps.CONST:
      self.lower_inputs(inputs)
      return
    if op is LoadOps.COPY:
      lowered_ins = self.lower_inputs(inputs)
      assert len(lowered_ins) == 1, f"only one input to copy allowed, given {len(lowered_ins)}"
      out_buff = self.lower_copy(output)
      in_buff = lowered_ins[0]
      # just an assign so no real addr needed
      addr = self.g.address(0,0,0)
      self.g.store(out_buff, addr, in_buff)
      return
    if op in BinaryOps:
      # Marshall the buffer views
      vt0, vt1, vt2 = inputs[0].vt, inputs[1].vt, output.vt
      # Strides may be different due to broadcasting
      strd0, strd1, strd2 = vt0.strides, vt1.strides, vt2.strides

      # Dimension and shapes should be the same
      # so use vt0
      loops, idxs = self.lower_start_loops(vt0.ndim, vt0.shape)
      self.lower_bop(inputs[0], inputs[1], output, idxs, strd0, strd1, strd2, 1, op)
      self.lower_end_loops(loops)
      return
    if op in TernaryOps:
      return 
    if op in ReduceOps:
      return
    raise NotImplementedError(f"lowering {op} is not supported")
  
  def lower(self):
    for fused_kernel in self.fused_kernels:
      if len(fused_kernel.computation.ops) == 1: 
        print("Single Op Lowering")
        self.lower_single_op_kernel(fused_kernel)
      else:
        raise NotImplementedError("Can't lower fused kernels yet")


        

