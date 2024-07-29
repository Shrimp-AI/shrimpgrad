
from enum import Enum, auto
from typing import Optional, Type, Union

class UnaryOps(Enum): EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); NEG = auto()
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPEQ = auto(); XOR = auto()
class TernaryOps(Enum): WHERE = auto(); MULACC = auto()
class ReduceOps(Enum): SUM = auto(); MAX = auto()
class BufferOps(Enum): LOAD = auto(); CONST = auto(); STORE = auto()
class LoadOps(Enum): EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); PAD = auto(); ASSIGN = auto()

Op = Union[UnaryOps, BinaryOps, ReduceOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[LoadOps], Type[TernaryOps], Type[BufferOps]]

# Fusion for fuse_ops.py requires these types
class AlgebraicOp(Enum): INJECTIVE = auto(); REDUCTION = auto(); NOOP = auto()

def injective(op: Op) -> bool: return op in BinaryOps or op in UnaryOps or op in TernaryOps
def reduction(op: Op) -> bool: return op in ReduceOps

def algebraic_op(op: Optional[Op]) -> AlgebraicOp:
  if op is None: return AlgebraicOp.NOOP
  if injective(op): return AlgebraicOp.INJECTIVE
  if reduction(op): return AlgebraicOp.REDUCTION
  return AlgebraicOp.NOOP
