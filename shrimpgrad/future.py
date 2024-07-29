from __future__ import annotations
from collections import defaultdict, deque
from functools import partial
from typing import Sequence, Callable, DefaultDict, List, Optional, Set, Tuple, TypeAlias, Union
from shrimpgrad.device import CPU, Device, Buffer, MemBuffer
from shrimpgrad.dtype import ConstType, DType
from shrimpgrad.runtime.ops import AlgebraicOp, BinaryOps, LoadOps, Op, ReduceOps, TernaryOps, UnaryOps, algebraic_op
from shrimpgrad.util import prod
from shrimpgrad.view import ViewTracker

def create_thunk(device: Device,
                 dtype: DType, vt: ViewTracker,
                 operands: Tuple[Thunk, ...]=(), op: Optional[Op]=None,
                 base: Optional[Thunk]=None, arg=None):
  return Thunk(device, dtype, vt, operands, op, base, arg)

class Thunk:
  """A lazy evaluated buffer described by future computation or memory operation.
  Thunks are realized only when they are needed for an operation. Thunks form an abstract
  syntax tree where root thunks have allocated buffers (inputs usually from tensor factory methods)
  and child thunks are described by operations and their parent operands.
  """
  def __init__(self, device: Device, dtype: DType, vt: ViewTracker, operands: Tuple[Thunk, ...]=(), op: Optional[Op]=None, base: Optional[Thunk]=None, arg=None):
    # initial buffer conditions
    self.vt , self.device, self.dtype, self.arg = vt, device, dtype, arg
    self._op, self._operands = op, operands

    # Am I the base thunk? Does the buffer reside with me or another thunk? Am I a view of the base thunk after a movement op? (same questions)
    # The base has no base because it is the base (so based)
    self._base = None
    if base is None:
      if op == LoadOps.ASSIGN:
        lhs = self._operands[0].base
        # += etc. requires lhs to be filled with the values of lhs
        self.buff = lhs.buff
      else:
        # I'm the base I own the real buffer
        self.buff = Buffer(self.device, self.vt.numel if not self.vt.scalar else 1, self.dtype)
    else:
      assert base.base == base, "base must be the base"
      self._base = base

  # Graph properties
  @property
  def parents(self) -> List[Thunk]:
    if self.isroot: return []  
    return [parent if not parent.isview else parent.base for parent in self._operands]
  @property
  def isroot(self) -> bool: return not hasattr(self, '_operands')
  @property
  def isview(self) -> bool: return self._op is None
  @property
  def isload(self) -> bool: return self._op in LoadOps if not self.isview else False
  @property
  def algebraic_op(self) -> AlgebraicOp: return algebraic_op(self._op)

  # Data properties
  @property
  def shape(self): return self.vt.shape
  @property
  def numel(self): return self.vt.numel
  @property
  def scalar(self): return self.vt.scalar
  @property
  def ndim(self): return self.vt.ndim
  @property
  def base(self): return self._base if self._base is not None else self
  @property
  def realized(self) -> Optional[Buffer]: return self.base.buff if hasattr(self.base, 'buff') and self.base.buff.allocated else None
  @property
  def strides(self): return self.vt.strides
  @property
  def isreduce(self): return self._op in ReduceOps
  @property
  def reduce_input_shape(self):
    assert(self.isreduce), f"{self._op} don't have a reduce input shape"
    return self._operands[0].shape

  def get_input_buffers(self) -> Sequence[MemBuffer]:
    inputs = []
    if self.isroot: return inputs
    for operand in self._operands:
      base = operand
      new_view = operand.vt
      if operand.isview: base = operand.base
      inputs.append(MemBuffer(base.buff, new_view))
    return inputs

  def get_output_buffer(self) -> MemBuffer:
    return MemBuffer(self.base.buff, self.vt)

  # Builder methods
  @staticmethod
  def from_compute(op: Union[BinaryOps, UnaryOps, TernaryOps, ReduceOps], operands: Tuple[Thunk,...], vt: ViewTracker, device: Device, dtype: DType) -> Thunk:
    if op in BinaryOps: assert len(operands) > 1, f'binary ops require two operands, {len(operands)} given'
    if op in UnaryOps: assert len(operands) > 0, f'unary ops require one operands, {len(operands)} given'
    if op in TernaryOps: assert len(operands) > 2, f'ternary ops require three operands, {len(operands)} given'
    return create_thunk(device, dtype, vt, tuple(operands), op)

  def alu(self, op: Union[UnaryOps, BinaryOps, TernaryOps], *in_thunks: Thunk) -> Thunk:
    return Thunk.from_compute(op, (self, *in_thunks), ViewTracker.from_shape(self.vt.shape), self.device, self.dtype)

  def reduce(self, op: ReduceOps, axis: Tuple[int,...]) -> Thunk:
    new_shape = tuple([1 if i in axis else s for i,s in enumerate(self.shape)])
    return create_thunk(self.device, self.dtype, ViewTracker.from_shape(new_shape), (self,), op, arg=axis)

  def reshape(self, shape: Tuple[int,...]) -> Thunk:
    return create_thunk(self.device, self.dtype, self.vt.reshape(shape), (), base=self.base)

  def permute(self, order:Tuple[int,...]) -> Thunk:
    return create_thunk(self.device, self.dtype, self.vt.permute(order), (), base=self.base)

  def expand(self, shape: Tuple[int,...]) -> Thunk:
    return create_thunk(self.device, self.dtype, self.vt.expand(shape), (), base=self.base)
  
  def pad(self, pad_width: Tuple[Tuple[int, int],...], value: ConstType=0.0):
    return create_thunk(self.device, self.dtype, self.vt.pad(pad_width), (), base=self.base, arg=value)

  def shrink(self, shrink_width: Tuple[Tuple[int, int],...]):
    return create_thunk(self.device, self.dtype, self.vt.shrink(shrink_width), (), base=self.base)

  def cast(self, dtype: DType) -> Thunk:
    return create_thunk(self.device, dtype, self.vt, (self,))
  
  @staticmethod
  def load_const(val: ConstType, shape: Tuple[int,...], dtype: DType, device: Device):
    assert isinstance(val, ConstType), f'load_const expects const val, got {val}'
    thunk =  Thunk.loadop(LoadOps.CONST, (), dtype, device, arg=val) 
    # Allocate so we don't schedule the load
    thunk.buff.allocate(with_data=val)
    return thunk.reshape((1,)*len(shape)).expand(shape)

  @staticmethod
  def load_from_cpu(data, dtype, shape):
    if not len(shape):
      assert isinstance(data, ConstType), 'scalar thunk requires a const argument'
      thunk = Thunk.loadop(LoadOps.CONST, shape, dtype, CPU(), arg=data)
      thunk.buff.allocate(with_data=data)
      return thunk
    assert len(data) == prod(shape), f'data and size mismatch {len(data)} != {prod(shape)}'
    thunk = Thunk.loadop(LoadOps.EMPTY, shape, dtype, CPU())
    thunk.buff.allocate(with_data=data)
    del thunk._operands
    return thunk

  @staticmethod
  def loadop(op: LoadOps, shape, dtype, device, arg=None, srcs=()):
    return create_thunk(device, dtype, ViewTracker.from_shape(shape), srcs, op=op, arg=arg)

  def copy_to_device(self, device: Device) -> Thunk:
    if self._op == LoadOps.CONST:
      self.device = device
      return self
    # Generaly self is a LoadOps.EMPTY with device as CPU
    # It may have been reshaped etc so ensure we copy from the base
    return create_thunk(device, self.dtype, self.vt, (self.base, ), LoadOps.COPY, arg=self.base.buff.nbytes)

  def const(self, val: ConstType, shape: Optional[Tuple[int,...]]=None):
    shape = self.shape if shape is None else shape
    thunk =  Thunk.loadop(LoadOps.CONST, (), self.dtype, self.device, arg=val) 
    thunk.buff.allocate(with_data=val)
    return thunk.reshape((1,)*len(shape)).expand(shape)

  def assign(self, rhs: Thunk) -> Thunk:
    assert self.numel == rhs.numel, f"assign size mismatch {self.numel} != {rhs.numel}"
    return Thunk.loadop(LoadOps.ASSIGN, self.shape, self.dtype, self.device, () if rhs.vt.contiguous else (rhs.vt,), (self.base, rhs))

  def contiguous(self) -> Thunk:
    if not self.vt.contiguous or self.base._op is LoadOps.CONST:
      return Thunk.loadop(LoadOps.CONTIGUOUS, self.shape, self.dtype, self.device, self.base.arg, (self,))
    return self

  def __str__(self) -> str: return f"<THUNK id={id(self)} {self.device} {self.vt} {str(self.dtype)[7:]} {self._op}>"
  def __repr__(self) -> str: return f"<THUNK {self._op} id={id(self)}>"
  def __hash__(self): return id(self)

# Graph Construction and Traversal helpers

###############################################################################
## Indexed Forward Graphs - Forward CFG of the output Thunk ###################
###############################################################################
SearchStrategy: TypeAlias = Callable[[Thunk], List[Thunk]]
SuccessorFn: TypeAlias = Callable[[Thunk], List[Thunk]]
Traversal: TypeAlias = Callable[[SearchStrategy, Thunk, SuccessorFn], List[Thunk]]

ThunkGraph: TypeAlias = DefaultDict[Thunk, List[Thunk]]
class IndexedForwardGraph:
  # Defaults to post order traversal
  def __init__(self, out: Thunk, traversal_fn='post'):
    self.out = out
    self.G: ThunkGraph = defaultdict(list)
    self.traversal_fn = self.post_order
    self.search_fn = lambda self: self._dfs()
    self.successor_fn = lambda thunk: [p for p in thunk._operands if hasattr(p, '_operands')]
    self.ordering = deque()
    self.node_to_num = {}
    # Traverse
    self.search_fn(self)

  def _dfs(self):
    def dfs(visited: Set[Thunk], thunk: Thunk):
      if thunk in visited: return
      visited.add(thunk.base) if thunk.isview else visited.add(thunk)
      self.traversal_fn(partial(dfs, visited), thunk) # type: ignore
    dfs(set(), self.out)

  def post_order(self, search: SearchStrategy, node: Thunk):
    if node.isview:
      node = node.base
    for succ in self.successor_fn(node):
      search(succ)
      self.G[succ].append(node)
    self.ordering.appendleft(node)
    self.node_to_num[node] = len(self.ordering) - 1

  def node2num(self, node: Thunk):
    return len(self.ordering) - 1 - self.node_to_num[node]