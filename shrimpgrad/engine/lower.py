from __future__ import annotations
from ctypes import Union
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Sequence, Tuple
from shrimpgrad.dtype import ConstType, DType
from shrimpgrad.runtime.ops import BinaryOps, TernaryOps, UnaryOps

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

  def address(self, idx: int, stride: int, step: int):
    self.G.append(node:=AddressNode(LowIR.ADDRESS, (), idx, stride, step))
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

  def begin_loop(self, start: ConstNode, end: ConstNode):
    self.G.append(node:=BeginLoopNode(LowIR.BEGIN_LOOP, (start, end)))
    return node

  def end_loop(self, loop: BeginLoopNode):
    self.G.append(node:=EndLoopNode(LowIR.END_LOOP, (loop, )))
    return node