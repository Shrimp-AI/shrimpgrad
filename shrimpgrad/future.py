from __future__ import annotations
from collections import defaultdict, deque
from functools import partial
from typing import Callable, DefaultDict, List, Optional, Set, Tuple, TypeAlias, Union
from shrimpgrad.device import CPU, Device, Buffer
from shrimpgrad.dtype import ConstType, DType 
from shrimpgrad.runtime.ops import AlgebraicOp, BinaryOps, LoadOps, Op, ReduceOps, TernaryOps, UnaryOps, algebraic_op
from shrimpgrad.util import prod
from shrimpgrad.view import View


# Thunk
#   - shapetracker
#      - list of views (each view has shape, strides, contiguous(are the strides swapped?), offset?, mask (for padding)?)
#      - the shapetracker consolidates all the movement ops on a thunk
#      - some thunks will have an st with multiple views (ex. permute(1,0).reshape((..)) b/c the permute makes the view non-contiguous
#        just reshaping after will lose the effect the permute had on the tensor
#   - base_thunk (the thunk that has the real buffer with data)
#     - a newly loaded thunk will own the base buffer
#     - movement ops merge with the st.views[-1] view if possible and return a new shapetracker which in turn 
#       creates a new thunk (which is a view b/c it's self.base != self)
class Thunk:
  """A lazy evaluated buffer described by future computation or memory operation.
  Thunks are realized only when they are needed for an operation. Thunks form an abstract
  syntax tree where root thunks have allocated buffers (inputs usually from tensor factory methods)
  and child thunks are described by operations and their parent operands.
  """
  def __init__(self, device: Device, dtype: DType, view: View, operands: Tuple[Thunk, ...]=(), op: Optional[Op]=None, data: Union[ConstType, List, bytes, memoryview]=None, base: Optional[Thunk]=None, arg=None):
    # initial buffer conditions
    self._view, self.device, self.dtype, self.arg = view, device, dtype, arg
    self._op, self._operands = op, operands 
    
    # Am I the base thunk? Does the buffer reside with me or another thunk? Am I a view of the base thunk after a movement op? (same questions)
    # The base has no base because it is the base (so based)
    self._base = None
    if base is None:
      if op != LoadOps.CONST:
        # I'm the base I own the real buffer 
        self.buff = Buffer(self.device, self._view.numel, self.dtype)
    else:
      assert base.base == base, "base must be the base"
      self._base = base
  
  # Graph properties
  @property
  def parents(self) -> Tuple[Thunk, ...]:
    if self.isroot: return () 
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
  def shape(self): return self._view.shape
  @property
  def numel(self): return self._view.numel
  @property
  def scalar(self): return self._view.scalar
  @property
  def ndim(self): return self._view.ndim
  @property
  def base(self): return self._base if self._base is not None else self
  @property
  def realized(self) -> Optional[Buffer]: return self.buff if hasattr(self, 'buff') and self.buff.allocated else None
  @property
  def strides(self): return self._view.strides 
  @property
  def isreduce(self): return self._op in ReduceOps
  @property
  def reduce_input_shape(self): 
    assert(self.isreduce), f"{self._op} don't have a reduce input shape"
    return self._operands[0].shape

  # Builder methods
  @staticmethod
  def from_compute(op: Union[BinaryOps, UnaryOps, TernaryOps, ReduceOps], operands: Tuple[Thunk,...], view: View, device: Device, dtype: DType):
    if op in BinaryOps: assert len(operands) > 1, f'binary ops require two operands, {len(operands)} given' 
    if op in UnaryOps: assert len(operands) > 0, f'unary ops require one operands, {len(operands)} given'
    if op in TernaryOps: assert len(operands) > 2, f'ternary ops require three operands, {len(operands)} given' 
    return Thunk(device, dtype, View.from_view(view), tuple(operands), op)

  def alu(self, op: Union[UnaryOps, BinaryOps, TernaryOps], *in_thunks: Tuple[Thunk,...]) -> Thunk:
    return Thunk.from_compute(op, (self, *in_thunks), self._view, self.device, self.dtype)

  def reduce(self, op: ReduceOps, axis: Tuple[int,...]) -> Thunk: 
    new_shape = tuple([1 if i in axis else s for i,s in enumerate(self._view.shape)])
    return Thunk(self.device, self.dtype, View(new_shape), (self,), op, arg=axis)

  def reshape(self, shape: Tuple[int,...]):
    return Thunk(self.device, self.dtype, self._view.reshape(shape), (), base=self.base)

  def permute(self, order:Tuple[int,...]) -> Thunk:
    return Thunk(self.device, self.dtype, self._view.permute(order), (), base=self.base)

  def expand(self, shape: Tuple[int,...]) -> Thunk:
    return Thunk(self.device, self.dtype, self._view.expand(shape), (), base=self.base)

  def cast(self, dtype: DType) -> Thunk:
    return Thunk(self.device, dtype, self._view, (self,))

  @staticmethod 
  def load_from_cpu(data, dtype, shape):
    if not len(shape):
      assert isinstance(data, ConstType), 'scalar thunk requires a const argument'
      return Thunk.loadop(LoadOps.CONST, shape, dtype, CPU(), arg=data) 
    thunk = Thunk.loadop(LoadOps.EMPTY, shape, dtype, CPU())
    thunk.buff.allocate(with_data=data)
    # This ensures we schedule it early for realize
    del thunk._operands
    return thunk
  
  @staticmethod
  def loadop(op: LoadOps, shape, dtype, device, arg=None, srcs=()):
    return Thunk(device, dtype, View(shape), srcs, op=op, arg=arg)
  
  def copy_to_device(self, device: Device) -> Thunk:
    if self._op == LoadOps.CONST: return self
    # Generaly self is a LoadOps.EMPTY with device as CPU
    # It may have been reshaped etc so ensure we copy from the base
    return Thunk(device, self.dtype, self._view, (self.base, ), LoadOps.COPY, arg=self.base.buff.nbytes)

  def const(self, val: ConstType, shape: Tuple[int,...]=None):
    shape = self.shape if shape is None else shape
    return Thunk.loadop(LoadOps.CONST, tuple(), self.dtype, self.device, arg=val).reshape((1,)*len(shape)).expand(shape)

  def __str__(self) -> str: return f"<THUNK {self.device} {self.shape} {str(self.dtype)[7:]} {self._op}>"
  def __repr__(self) -> str: return f"<THUNK {self._op} id={id(self)}>"
  def __hash__(self): return id(self)

# Graph Construction and Traversal helpers 

###############################################################################
## Indexed Forward Graphs - Forward CFG of the output Thunk ###################
###############################################################################
# In order to minimize graph traversal we want to collect as much as we can
# in a single pass. Save conditions allow us to extract nodes from a graph into
# separate storage
# For example: Save if Load, Save if Expand, etc. will save those nodes even if
# the graph wont contain them (fusion doesn't require virtual nodes and loads)
# SearchMode is a dumb dfs or dumb bfs that requires successor function
# and traversal (manages a visited set and kicks off the recursion)
# SuccesorFn describes how to get the successor from a thunk
# Traversal is a way to construct an order of the DAG (uses successor function and dumbdfs)
SaveCondition:TypeAlias = Callable[[Thunk], bool]
IgnoreCondition: TypeAlias = Callable[[Thunk], bool]
SearchStrategy: TypeAlias = Callable[[Thunk], List[Thunk]] 
SuccessorFn: TypeAlias = Callable[[Thunk], List[Thunk]]
Traversal: TypeAlias = Callable[[SearchStrategy, Thunk, SuccessorFn], List[Thunk]]

save_loads: SaveCondition = lambda t: t.isload and not t._op == LoadOps.CONST
save_expands: SaveCondition = lambda t: t.isview and not t.isload and prod(t.base.shape) < prod(t.shape)
save_roots: SaveCondition = lambda t: t.isroot
save_const: SaveCondition = lambda t: t._op == LoadOps.CONST

ignore_load: IgnoreCondition = lambda thunk: thunk.isload
ignore_root: IgnoreCondition = lambda thunk: thunk.isroot
ignore_view: IgnoreCondition =lambda thunk: thunk.isview




ThunkGraph: TypeAlias = DefaultDict[Thunk, List[Thunk]]
class IndexedForwardGraph:
  # Defaults to post order traversal
  # Defaults to ignore loads and roots
  # Defaults to save loads, expands, roots, and const loads
  def __init__(self, out: Thunk,
               save_conditions: Optional[List[SaveCondition]]=None,
               ignore_conditions: Optional[List[IgnoreCondition]]=None,
               traversal_fn='post'):
    self.out = out
    self.G: ThunkGraph = defaultdict(list)
    self.traversal_fn = self.post_order 
    self.save_conditions = save_conditions if save_conditions is not None else [save_loads, save_expands, save_roots, save_const]
    self.saved: DefaultDict[int, DefaultDict[Thunk, Set[Thunk]]]= defaultdict(lambda: defaultdict(set))
    self.ignore_conditions = ignore_conditions if ignore_conditions is not None else [ignore_load, ignore_root] 
    self.search_fn = lambda self: self._dfs()
    self.successor_fn = lambda thunk: [p for p in thunk._operands if hasattr(p, '_operands')] 
    self.ordering = deque() 
    self.node_to_num = {}
    # Traverse
    self.search_fn(self)
  
  def save(self, thunk: Thunk, output: Thunk) -> None:
    for skey, scnd in enumerate(self.save_conditions): self.saved[skey][thunk].add(output) if scnd(thunk) else None 
  
  def ignore(self, thunk: Thunk) -> bool:
    return any([fn(thunk) for fn in self.ignore_conditions]) 

  def _dfs(self):
    def dfs(visited: Set[Thunk], thunk: Thunk):
      if thunk in visited: return
      visited.add(thunk.base) if thunk.isview else visited.add(thunk)
      self.traversal_fn(partial(dfs, visited), thunk)
    dfs(set(), self.out)

  def post_order(self, search: SearchStrategy, node: Thunk):
    if node.isview:
      node = node.base
    for succ in self.successor_fn(node):
      self.save(succ, node) 
      if (self.ignore(succ)): continue
      search(succ)
      self.G[succ].append(node)
    self.ordering.appendleft(node)
    self.node_to_num[node] = len(self.ordering) - 1
  
  def node2num(self, node: Thunk):
    return len(self.ordering) - 1 - self.node_to_num[node]