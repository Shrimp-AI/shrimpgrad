from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import functools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from shrimpgrad.device import MemBuffer
from shrimpgrad.dtype import ConstType, DType, dtypes
from shrimpgrad.engine.scheduler import FusedKernel
from shrimpgrad.runtime.ops import BinaryOps, LoadOps, ReduceOps, TernaryOps, UnaryOps
from shrimpgrad.util import prod
from shrimpgrad.view import ViewTracker

def alu2str(op:BinaryOps|UnaryOps|TernaryOps) -> str:
  assert op in BinaryOps or op in UnaryOps or op in TernaryOps, f"{op} is not a binary/unary/ternary alu op"
  if op is BinaryOps.ADD: return '+'
  if op is BinaryOps.CMPEQ: return '=='
  if op is BinaryOps.CMPLT: return '<'
  if op is BinaryOps.DIV: return '/'
  if op is BinaryOps.MAX: return 'max'
  if op is BinaryOps.MOD: return '%'
  if op is BinaryOps.MUL: return '*'
  if op is BinaryOps.SUB: return '-'
  if op is BinaryOps.XOR: return 'xor'
  if op is UnaryOps.CAST: return 'cast'
  if op is UnaryOps.EXP2: return 'exp2'
  if op is UnaryOps.LOG2: return 'log2'
  if op is UnaryOps.NEG: return '-'
  if op is UnaryOps.SIN: return 'sin'
  if op is UnaryOps.SQRT: return 'sqrt'
  if op is TernaryOps.WHERE: return 'where'
  raise ValueError(f"{op} is not a valid alu op")

class LowIR(Enum):
  # Variables
  GLOBAL = auto(); LOCAL = auto(); ACC = auto(); CONST=auto()
  # Indexing 
  ADDRESS = auto(); OFFSET = auto()
  # Memory load and store 
  LOAD = auto(); STORE = auto()
  # ALU
  INC = auto(); ALU = auto()
  # Control Flow
  BEGIN_LOOP = auto(); END_LOOP = auto()
  IF = auto(); ENDIF = auto()

@dataclass(frozen=True, eq=True)
class Node:
  op: LowIR
  ancestors: Tuple[Node|None,...]

@dataclass(frozen=True, eq=True)
class ConstNode(Node):
  dtype: DType
  val: ConstType
  def __repr__(self) -> str: return f"{'CONST':<15}{self.val}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class GlobalNode(Node):
  name: str
  dtype: DType
  ptr: bool
  pos: int
  mutable: bool
  def __repr__(self) -> str:
    return f"{'DEFINE_GLOBAL':<15}{self.name:<10}{('ptr.'+str(self.dtype) if self.ptr else str(self.dtype)):<10}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class LocalNode(Node):
  name: str
  dtype: DType
  def __repr__(self) -> str:
    assert self.ancestors[0] is not None, "local cant be None"
    return f"{'DEFINE_LOCAL':<15}{self.name:<10}{self.ancestors[0].op:<10}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class AddressNode(Node):
  idx: Tuple[LocalNode|int,...]
  stride: Tuple[int,...]
  step: int
  vt: Optional[ViewTracker]=None
  def __repr__(self) -> str:
    if self.vt is not None:
      return f"{'ADDRESS':<15}{self.vt.render()[0]:<10}"
    addr = ''
    for idx, stride in zip(self.idx, self.stride):
      val = idx.name if isinstance(idx, LocalNode) else idx
      addr += f"{val}*{stride}*{self.step}+"
    return f"{'ADDRESS':<15}{addr[:-1]:<10}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class LoadNode(Node):
  def __repr__(self) -> str:
    assert self.ancestors[1] is not None 
    op = '' if not self.ancestors[0].ptr else str(self.ancestors[1].op)
    return f"{'LOAD':<15}{self.ancestors[0].name:<10}{str(self.ancestors[0].dtype):<20}{op:<10}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class AccumulatorNode(Node):
  name: str
  dtype: DType
  alu: BinaryOps
  def __repr__(self) -> str:
    return f"{'ACC':<15}{self.alu}{self.ancestors}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class StoreNode(Node):
  def __repr__(self) -> str:
    addr = str(self.ancestors[1].op) if self.ancestors[1] is not None else "noop"
    return f"{'STORE':<15}{self.ancestors[0].name:<10}{addr:<20}{str(self.ancestors[2].op):<10}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class ALUNode(Node):
  alu: BinaryOps|UnaryOps|TernaryOps
  dtype: DType
  def __repr__(self) -> str:
    operands = f"{' ':<10}".join([f"{str(operand.op)}" for operand in self.ancestors])
    return f"{'ALU':<15}{alu2str(self.alu):<9} {operands}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class BeginLoopNode(Node):
  def __repr__(self) -> str:
    return f"{'BEGIN_LOOP':<15}{self.ancestors[0].name:<10}{str(self.ancestors[1].val):<10}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class EndLoopNode(Node):
  def __repr__(self) -> str:
    return "END_LOOP"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class OffsetNode(Node):
  def __repr__(self) -> str:
    return  f"{'OFFSET':<15}{str(self.ancestors[0])[15:]:<10}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

@dataclass(frozen=True, eq=True)
class IncNode(Node):
  def __repr__(self) -> str:
    return f"{'INC':<15}{self.ancestors[0].name:<10}"
  @functools.cached_property
  def hash(self): return hash((self.op, self.ancestors))
  def __hash__(self): return self.hash

# A graph where each node occupies an index (based on the order of addition)
# and has 0-to-Many back pointers to dependecies via node.ancestors
class LowIRGraph:
  def __init__(self):
    self.G: List[Node] = []

  def const(self, dtype: DType, val: ConstType) -> ConstNode:
    self.G.append(node:=ConstNode(LowIR.CONST, (), dtype, val))
    return node

  def define_global(self, name:str, dtype: DType, mutable: bool, pos: int, ptr: bool=True) -> GlobalNode:
    self.G.append(node:=GlobalNode(LowIR.GLOBAL, (), name, dtype, ptr, pos, mutable))
    return node

  def local_var(self, name: str, dtype: DType, val: ConstNode|ALUNode) -> LocalNode:
    self.G.append(node:=LocalNode(LowIR.LOCAL, (val, ), name, dtype))
    return node

  def address(self, idxs: List[LocalNode] | List[int], strides: Tuple[int,...], step: int, vt: Optional[ViewTracker]=None) -> AddressNode:
    if vt is not None:
      self.G.append(node:=AddressNode(LowIR.ADDRESS, (), tuple(idxs), (), 0, vt))
      return node
    self.G.append(node:=AddressNode(LowIR.ADDRESS, (), tuple(idxs), strides, step))
    return node

  def load(self, inp: GlobalNode|LocalNode, location: Optional[AddressNode|OffsetNode]) -> LoadNode:
    self.G.append(node:=LoadNode(LowIR.LOAD, (inp, location)))
    return node

  def accumulator(self, alu: BinaryOps,
                  name: str, dtype: DType,
                  operands: Sequence[ConstNode|LoadNode]) -> AccumulatorNode:
    self.G.append(node:=AccumulatorNode(LowIR.ACC, tuple(operands), name, dtype, alu))
    return node

  def store(self, lhs: GlobalNode|LocalNode,
            address: AddressNode|OffsetNode|None|ViewTracker,
            rhs: LoadNode|ConstNode|LocalNode|ALUNode) -> StoreNode:
    self.G.append(node:=StoreNode(LowIR.STORE, (lhs, address, rhs)))
    return node

  def alu(self, alu: BinaryOps|UnaryOps|TernaryOps, dtype: DType, *operands) -> ALUNode:
    self.G.append(node:=ALUNode(LowIR.ALU, tuple(operands), alu, dtype))
    return node

  def begin_loop(self, start: LocalNode, end: ConstNode) -> BeginLoopNode:
    self.G.append(node:=BeginLoopNode(LowIR.BEGIN_LOOP, (start, end)))
    return node

  def end_loop(self, loop: BeginLoopNode) -> EndLoopNode:
    self.G.append(node:=EndLoopNode(LowIR.END_LOOP, (loop, )))
    return node

  def offset(self, val: ConstNode|LocalNode|ALUNode) -> OffsetNode:
    self.G.append(node:=OffsetNode(LowIR.OFFSET, (val,)))
    return node

  def inc(self, var: LocalNode) -> IncNode:
    self.G.append(node:=IncNode(LowIR.INC, (var,)))
    return node

  def __str__(self) -> str:
    return "\n".join([str(node) for node in self.G])

  def print(self):
    for node in self.G:
      print(node)

class LowerFusedKernel:
  def __init__(self, fused_kernels: List[FusedKernel]):
    self.fused_kernels = fused_kernels
    self.symbol_table: Dict[str, Node] = {}
    self.node_to_symbol: Dict[Any, str] = {}
    self.arg_count = 0
    self.global_input_counter = 0
    self.global_output_counter = 0
    self.local_counter = 0
    self.accum_counter = 0
    self.g = LowIRGraph()
    # TODO: Get dtype from inputs or outputs
    self.dtype = dtypes.float32
    self.consts = []

  def global_name(self, prefix="data"):
    if prefix == "data":
      val = self.global_input_counter
      self.global_input_counter += 1
    else:
      val = self.global_output_counter
      self.global_output_counter += 1
    name = f"{prefix}{val}"
    self.arg_count += 1
    return name

  def local_name(self, prefix="idx"):
    name = f"{prefix}{self.local_counter}"
    self.local_counter += 1
    return name

  def accum_name(self):
    name = f"acc{self.accum_counter}"
    self.accum_counter += 1
    return name

  # Lower a buffer input or output into a global
  def lower_buffer(self, mbuff: MemBuffer, is_input: bool) -> GlobalNode:
    if mbuff in self.node_to_symbol: return cast(GlobalNode, self.symbol_table[self.node_to_symbol[mbuff]])
    prefix, mutable = ("out", True) if not is_input else ("data", False)
    g0 = self.g.define_global(name:=self.global_name(prefix), mbuff.buff.dtype, mutable, self.arg_count - 1, True)
    self.symbol_table[name] = g0
    self.node_to_symbol[mbuff] = name
    return g0

  def lower_io(self, io: MemBuffer, is_input:bool) -> GlobalNode:
    return self.lower_buffer(io, is_input)

  def lower_load(self, g: GlobalNode, addr: Optional[AddressNode|OffsetNode]=None) -> LoadNode:
    # If g is a pointer, load with an address, otw. just load the value
    return self.g.load(g, addr) if g.ptr else self.g.load(g, None)

  def lower_alu(self, alu: BinaryOps|TernaryOps|UnaryOps, *loads) -> ALUNode:
    # TODO: Deal with dtype
    return self.g.alu(alu, self.dtype, *loads)

  def lower_store(self, g: GlobalNode|LocalNode, addr: AddressNode|OffsetNode|None|ViewTracker, value: LoadNode|ConstNode|LocalNode|ALUNode) -> StoreNode:
    if g.__class__ is LocalNode: return self.g.store(g, None, value)
    return self.g.store(g, addr, value)

  def lower_local(self, dtype: DType, val: ConstType|ALUNode) -> LocalNode:
    return self.g.local_var(self.local_name(), dtype, self.g.const(dtype, val) if not isinstance(val, ALUNode) else val)
  
  def lower_acc(self, dtype: DType, val: ConstType|ALUNode) -> LocalNode:
    return self.g.local_var(self.accum_name(), dtype, self.g.const(dtype, val) if not isinstance(val, ALUNode) else val)

  def lower_bop(self,
                in0: MemBuffer,
                in1: MemBuffer,
                out0:MemBuffer,
                idxs: List[LocalNode],
                strides0: Tuple[int,...],
                strides1: Tuple[int,...],
                strides2: Tuple[int,...],
                step: int,
                alu: BinaryOps):
    # Define a global for the output
    out = self.lower_io(out0, False)

    # Load the two inputs
    g0 = self.lower_io(in0, True)
    addr0 = self.g.address(idxs, strides0, step)
    l0 = self.lower_load(g0, addr0)

    g1 = self.lower_io(in1, True)
    addr1 = self.g.address(idxs, strides1, step)
    l1 = self.lower_load(g1, addr1)

    #  Lower the binary ALU operation
    alu0 = self.lower_alu(alu, l0, l1)

    addr2 = self.g.address(idxs, strides2, step)
    self.lower_store(out, addr2, alu0)

  def lower_uop(self,
                in0: MemBuffer,
                out0: MemBuffer,
                idxs: List[LocalNode],
                strd0: Tuple[int,...],
                strd1: Tuple[int,...],
                step: int,
                alu: UnaryOps):
    # Define a global for the output
    out = self.lower_io(out0, False)

    g0 = self.lower_io(in0, is_input=True)
    addr0 = self.g.address(idxs, strd0, step)
    l0 = self.lower_load(g0, addr0)


    alu0 = self.lower_alu(alu, l0)
    addr2 = self.g.address(idxs, strd1, step)
    self.lower_store(out, addr2, alu0)

  def lower_top(self, 
                cond: MemBuffer,
                a: MemBuffer,
                b: MemBuffer,
                out: MemBuffer,
                idxs: List[LocalNode],
                vt0: ViewTracker, vt1: ViewTracker, vt2: ViewTracker, vtout: ViewTracker):
    # Define a global for the output
    gout = self.lower_io(out, False)
    gcond = self.lower_io(cond, True)
    ga = self.lower_io(a, True)
    gb = self.lower_io(b, True)
    addr0 = self.g.address(idxs, (), 0, vt0)
    addr1 = self.g.address(idxs, (), 0, vt1)
    addr2 = self.g.address(idxs, (), 0, vt2)
    lc = self.lower_load(gcond, addr0)
    la = self.lower_load(ga, addr1)
    lb = self.lower_load(gb, addr2)
    alu0 = self.lower_alu(TernaryOps.WHERE, lc, la, lb)
    addr3 = self.g.address(idxs, (), 0, vtout)
    self.lower_store(gout, addr3, alu0)

  def lower_rop(self,
                in0: MemBuffer,
                out0: MemBuffer ,
                axis: Tuple[int, ...],
                rop: ReduceOps):
    in_shape = in0.vt.shape
    out_shape = out0.vt.shape
    out = self.lower_io(out0, False)
    # Define a global for the input
    g_in = self.lower_io(in0, True)
    rdtype = in0.buff.dtype
    alu_op, acc_init = (BinaryOps.ADD, 0.0) if rop is ReduceOps.SUM else (BinaryOps.MAX, -math.inf)
    if len(axis) == len(in_shape) and in0.vt.contiguous:
      zero = self.g.const(dtypes.int32, 0)
      out_off = self.g.offset(zero)
      acc = self.lower_acc(rdtype, acc_init)
      loop, idx = self.lower_begin_for(0, prod(in0.vt.shape))
      in_off = self.g.offset(idx)
      rhs = self.lower_load(g_in, idx)
      alu = self.lower_alu(alu_op, acc, rhs)
      self.lower_store(acc, None, alu)
      self.lower_end_loops([loop])
      self.lower_store(out, out_off, acc)
    else:
      # Move reduce axes to the end via creating a permutation order
      order = tuple([i for i,_ in enumerate(in_shape) if in_shape[i] == out_shape[i]] + [i for i,_ in enumerate(in_shape) if out_shape[i] != in_shape[i]])
      in0_vt = in0.vt.permute(order)
      in_shape = in0_vt.shape
      # add an output offset
      off = self.lower_local(dtypes.int32, 0)
      num_axis = len(axis)
      acc = None
      # Create a loop for each dimension that's not the last dimension
      loops, idxs, dim_offs = [], [], []
      for i in range(len(in_shape) - 1):
        loop, idx = self.lower_begin_for(0, in_shape[i])
        if i+1 == len(in_shape) - num_axis: acc = self.lower_acc(rdtype, acc_init)
        idxs.append(idx)
        loops.append(loop)
        # Compute the dimension offset l0*strides[i]
        alu = self.lower_alu(BinaryOps.MUL, idx, self.g.const(dtypes.int32, in0_vt.strides[i]))
        # Set the dim off set to the alu value
        dim_off = self.lower_local(dtypes.int32, alu)
        dim_offs.append(dim_off)
      if acc is None:
        acc = self.lower_acc(rdtype, acc_init)
      # Create the inner loop
      loop, idx = self.lower_begin_for(0, in_shape[-1])
      idxs.append(idx)
      loops.append(loop)

      # Multiply the inner loop idx with the final stride
      alu1 = self.lower_alu(BinaryOps.MUL, idx, self.g.const(dtypes.int32, in0_vt.strides[-1]))
      # Sum all the offsets to get the true input offset
      # Non contiguous full axis reduces with one dimension will not havre
      # dim offs
      if not dim_offs:
        alu2 = alu1
      else:
        alu2 = self.lower_alu(BinaryOps.ADD, *dim_offs, alu1)
      in_off = self.lower_local(dtypes.int32, alu2)
      # Accumlate in out
      out_off = self.g.offset(off)
      in_off = self.g.offset(in_off)
      rhs = self.lower_load(g_in, in_off)
      alu = self.lower_alu(alu_op, acc, rhs)
      self.lower_store(acc, None, alu)

      for i, loop in enumerate(loops):
        self.lower_end_loops([loop])
        if i + 1 == num_axis:
          self.lower_store(out, out_off, acc)
          self.g.inc(off)

  def lower_begin_for(self, s: int, e: int) -> Tuple[BeginLoopNode, LocalNode]:
    c0 = self.g.const(dtypes.int32, s) # start
    c1 = self.g.const(dtypes.int32, e) # end
    # Loop index
    idx = self.g.local_var(self.local_name(), dtypes.int32, c0)
    # Begin a loop from var = 0 to c1
    loop = self.g.begin_loop(idx, c1)
    return loop, idx

  # TODO: Loop unrolling (but not ncessary once we gen for GPU)
  def lower_start_loops(self, ndim:int, shape: Tuple[int,...]) -> Tuple[List[BeginLoopNode], List[LocalNode]]:
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
    for loop in loops:
      self.g.end_loop(loop)

  def lower_assign(self, out, rhs, vt_lhs, vt_rhs, idxs):
    strd0, strd1 = vt_lhs.strides, vt_rhs.strides
    if not idxs:
      lrhs = self.lower_load(rhs, None)
      self.lower_store(out, None, lrhs)
      return
    addr0 = self.g.address(idxs, strd0, 1)
    addr2 = self.g.address(idxs, strd1, 1)
    lrhs = self.lower_load(rhs, addr2)
    self.lower_store(out, addr0, lrhs)
    return

  def lower_single_op_kernel(self, fused_kernel: FusedKernel):
    assert len(fused_kernel.computation.ins) == 1
    assert len(fused_kernel.computation.out) == 1
    inputs = fused_kernel.computation.ins[0]
    output = fused_kernel.computation.out[0]
    op = fused_kernel.computation.ops[0]
    arg = fused_kernel.computation.args[0]
    if op is LoadOps.CONST or op is LoadOps.CONTIGUOUS:
      gout = self.lower_io(output, is_input=False)
      gin = self.lower_io(inputs[0], True)
      loops, idxs = self.lower_start_loops(output.vt.ndim, output.vt.shape)
      rhs = self.lower_load(gin, self.g.address(idxs, (), 0, inputs[0].vt))
      addr = self.g.address(idxs, output.vt.strides, 1)
      self.lower_store(gout, addr, rhs) 
      self.lower_end_loops(loops)
      return
    if op in LoadOps and op is not LoadOps.ASSIGN:
      self.lower_io(output, is_input=True)
      return
    if op is LoadOps.ASSIGN:
      # input 0 is the same as output
      vt0, vt1 = inputs[0].vt, inputs[1].vt
      _ = self.lower_io(inputs[0], True)
      rhs = self.lower_io(inputs[1], True)
      out = self.lower_io(output, False)
      if vt0.scalar and vt1.scalar:
        self.lower_assign(out, rhs, vt0, vt1, [])
        return
      loops, idxs = self.lower_start_loops(vt0.ndim, vt0.shape)
      self.lower_assign(out, rhs, vt0, vt1, idxs)
      self.lower_end_loops(loops)
      return
    if op in BinaryOps:
      # Marshall the buffer views
      vt0, vt1, vt2 = inputs[0].vt, inputs[1].vt, output.vt
      # Strides may be different due to broadcasting
      strd0, strd1, strd2 = vt0.strides, vt1.strides, vt2.strides

      # Dimension and shapes should be the same
      # so use vt0 unless it's a const
      if inputs[0].vt.scalar:
        loops, idxs = self.lower_start_loops(vt1.ndim, vt1.shape)
      else:
        loops, idxs = self.lower_start_loops(vt0.ndim, vt0.shape)
      self.lower_bop(inputs[0], inputs[1], output, idxs, strd0, strd1, strd2, 1, op)
      self.lower_end_loops(loops)
      return
    if op in UnaryOps:
      vt0 = inputs[0].vt
      vt1 = output.vt
      strd0 = vt0.strides
      strd1 = vt1.strides
      loops, idxs = self.lower_start_loops(vt0.ndim, vt0.shape)
      self.lower_uop(inputs[0], output, idxs, strd0, strd1, 1, op)
      self.lower_end_loops(loops)
      return

    if op in TernaryOps:
      assert len(inputs) == 3, 'ternary op requires three inputs'
      cond, a, b = inputs[0], inputs[1], inputs[2]
      vt0 = cond.vt
      loops, idxs = self.lower_start_loops(vt0.ndim, vt0.shape)
      self.lower_top(cond, a, b, output, idxs, cond.vt, a.vt, b.vt, output.vt)
      self.lower_end_loops(loops)
      return
    if op in ReduceOps:
      self.lower_rop(inputs[0], output, arg, op)
      return
    raise NotImplementedError(f"lowering {op} is not supported")

  def lower_multi_op_kernel(self, fk: FusedKernel):
    ins, outs, ops, args = fk.computation.ins, fk.computation.out, fk.computation.ops, fk.computation.args
    assert len(ins) > 1, 'multi op lowering requires multi input'
    assert len(outs) > 1, 'multi op lowering requires multi output'
    assert len(ops) > 1, 'multi op lowering requires multi ops'
    assert len(args) > 1, ' multi op lowering requires multi args'
    assert len(ins) == len(outs) == len(ops) == len(args), 'multi op lowering requires inputs, outputs, ops, and args match in length'
    loops,idxs = None, None
    for i, o, op, arg in zip(ins, outs, ops, args):
      if not loops:
        vt0 = o.vt
        loops, idxs = self.lower_start_loops(vt0.ndim, vt0.shape)
      if op in BinaryOps:
        # Marshall the buffer views
        vt0, vt1, vt2 = i[0].vt, i[1].vt, o.vt
        # Strides may be different due to broadcasting
        strd0, strd1, strd2 = vt0.strides, vt1.strides, vt2.strides

        # Dimension and shapes should be the same
        # so use vt0
        self.lower_bop(i[0], i[1], o, idxs, strd0, strd1, strd2, 1, op)
        continue
      if op in UnaryOps:
        vt0 = i[0].vt
        vt1 = o.vt
        strd0 = vt0.strides
        strd1 = vt1.strides
        self.lower_uop(i[0], o, idxs, strd0, strd1, 1, op)
        continue
      if op in TernaryOps:
        vt0, vt1, vt2, vtout = i[0].vt, i[1].vt, i[2].vt, o.vt
        assert len(i) == 3, 'ternary op requires three inputs'
        cond, a, b = i[0], i[1], i[2]
        self.lower_top(cond, a, b, o, idxs, vt0, vt1, vt2, vtout)
        continue 
      if op in ReduceOps:
        assert len(i) == 1, "reduce only has 1 input"
        # Terminate the loops since reduce occurs after them
        self.lower_end_loops(loops)
        self.lower_rop(i[0], o, arg, op)
        # Return because reduce is always the last fused operation
        return
      if op is LoadOps.ASSIGN:
        # input 0 is the same as output
        _ = self.lower_io(i[0], True)
        rhs = self.lower_io(i[1], True)
        out = self.lower_io(o, False)
        self.lower_assign(out, rhs, o.vt, i[1].vt, idxs)
    self.lower_end_loops(loops)

  def lower(self) -> List[LowIRGraph]:
    graphs: List[LowIRGraph] = []
    for fused_kernel in self.fused_kernels:
      if len(fused_kernel.computation.ops) == 1:
        self.lower_single_op_kernel(fused_kernel)
      else:
        self.lower_multi_op_kernel(fused_kernel)
      graphs.append(self.g)
      self.local_counter = 0
      self.arg_count = 0
      self.g = LowIRGraph()
    return graphs