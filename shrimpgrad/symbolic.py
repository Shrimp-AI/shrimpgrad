from __future__ import annotations
from enum import Enum, auto
from typing import Optional, TypeAlias
from dataclasses import dataclass

@dataclass
class Interval: 
  start:symbolicint 
  end: symbolicint 


class Show:
  def show(self) -> str:
    raise NotImplementedError('implement show')

class ArithOp(Show, Enum):
  PLUS=auto()
  def show(self):
    if self.name == 'PLUS': return "+"
  def __str__(self):
    return self.name

class Expr(Show):
  pass

class Symbol(Expr):
  def __init__(self, name:str, domain: Optional[Interval]=None):
    self.name: str = name
    self.domain: Optional[Interval] = domain
  def __add__(self, other: Symbol):
    return Bin(ArithOp.PLUS, self, other)  
  def __repr__(self):
    return self.name
  def __str__(self):
    return self.name

class Lit(Expr):
  def __init__(self, val:int): self.val = val
  def __repr__(self): return f"(Lit {self.val})"
  def __str__(self): return str(self.val)

class Bin(Expr):
  def __init__(self, op, lhs: Expr, rhs: Expr):
    self.op, self.lhs, self.rhs = op, lhs, rhs
  def __repr__(self):
    return f"(Bin {self.op} {self.lhs} {self.rhs})"
  def __str__(self):
    return f"{self.lhs} {self.op.show()} {self.rhs}"

symbolicint: TypeAlias = Symbol|int