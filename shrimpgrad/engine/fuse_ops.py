from typing import List, Optional
from shrimpgrad.engine.postdomtree import ipdoms
from shrimpgrad.future import Thunk
from shrimpgrad.runtime.ops import AlgebraicOp 

class FusedOp:
  # Injective Operators are BinaryOps and UnaryOps and TernaryOps
  # Reduction are ReduceOps
  # Rules for Fusion:
  # 1. Injective operators can fuse with other injective operators
  # 2. Reduction operators can fuse with input injective operators to form a fused
  #    injective operator 

  # Fusion always starts with an injective
  # since we can't fuse multiple reductions
  def __init__(self, injective_thunk: Thunk):
    self.reduce: Optional[Thunk] = None
    self.injective: List[Thunk] = [injective_thunk]
    self.shape = injective_thunk.shape

  @property
  def has_reduce(self): return self.reduce is not None
  @property
  def shape(self): return self.shape

  def can_fuse(self, thunk: Thunk) -> bool:
    # Chained injectives preserve shape due to broadcasting
    # Reduction ops will have the input shape equal to the output shape of the previous injective op
    # if there was one. Axis will reduce the shape post execution hence we can fuse.
    if thunk.shape != self.shape: return False
    # Can always fuse injectives if the shapes match
    if thunk.algebraic_op is AlgebraicOp.INJECTIVE: return True
    # Can only fuse reductions if there is not a reduction in this FusedOp
    if thunk.algebraic_op is AlgebraicOp.REDUCTION and not self.has_reduce: return True
    # Can't fuse NOOPs (shouldn't be in the reduced reverse graph anyway but just in  case)
    return False

  # Returns True on fusion and False on non-fusion
  def fuse(self, thunk: Thunk) -> bool:
    if self.can_fuse(thunk):  
      if thunk.algebraic_op is AlgebraicOp.INJECTIVE: self.injective.append(thunk)
      else: self.reduce = thunk
      return True
    return False

class FusionEngine:
  def __init__(self, thunk: Thunk): 
    self.root = thunk
    self.ipdoms = ipdoms(self.root)
 
  def start(self):
    # Start fusing
    pass

  def _traverse_to_ipdom(self):
    # From every node traverse to it's IPDOM and make a group of potential fuses
    pass
