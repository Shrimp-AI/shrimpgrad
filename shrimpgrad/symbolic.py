from __future__ import annotations
from enum import Enum, auto
from typing import Optional, Tuple, TypeAlias
from dataclasses import dataclass

@dataclass
class Interval: 
  start:sint
  end: sint

class ArithOp(Enum):
  MUL=auto(); DIV=auto(); MOD=auto(); NEG=auto()
  PLUS=auto(); SUB=auto() 
  AND=auto(); OR=auto()
  LT=auto(); GT=auto(); LTE=auto(); GTE=auto(); EQ=auto()
  def show(self):
    if self == self.PLUS: return "+"
    if self == self.MUL: return "*"
    if self == self.SUB: return "-"
    if self == self.DIV: return  "/"
    if self == self.MOD: return "%"
    if self == self.LT: return "<"
    if self == self.GT: return ">"
    if self == self.LTE: return "<="
    if self == self.GTE: return ">="
    if self == self.EQ: return "="
    if self == self.AND: return "&&"
    if self == self.NEG: return "-"
    return "||"

precedence = {
  ArithOp.MUL: 4, ArithOp.DIV: 4, ArithOp.MOD: 4,
  ArithOp.PLUS: 3,ArithOp.SUB: 3,
  ArithOp.AND: 2, ArithOp.OR: 2,
  ArithOp.LT: 1, ArithOp.GT: 1, ArithOp.LTE: 1, ArithOp.GTE: 1, ArithOp.EQ: 1
}

class Expr:
  def __add__(self, other: Expr): return Bin(ArithOp.PLUS, self, other)  
  def __iadd__(self, other: Expr): return self + other
  def __radd__(self, other): return self + other 
  def __mul__(self, other: Expr): return Bin(ArithOp.MUL, self, other)
  def __rmul__(self, other: Expr): return self * other 
  def __sub__(self, other: Expr): return Bin(ArithOp.SUB, self, other) 
  def __truediv__(self, other: Expr): return Bin(ArithOp.DIV, self, other) 
  def __rtruediv__(self, other): return self /other 
  def __mod__(self, other: Expr): return Bin(ArithOp.MOD, self, other)
  def __rmod__(self, other: Expr): return self % other 
  def __lt__(self, other: Expr): return Bin(ArithOp.LT, self, other)
  def __le__(self, other: Expr): return Bin(ArithOp.LTE, self, other)
  def __gt__(self, other: Expr): return Bin(ArithOp.GT, self, other)
  def __ge__(self, other: Expr): return Bin(ArithOp.GTE, self, other)
  def __eq__(self, other: Expr): return Bin(ArithOp.EQ, self, other)
  def __neg__(self): return Unary(ArithOp.NEG, self) 
  def ifelse(self, x, y): return IfElse(self, x , y)
  def and_(self, x): return Bin(ArithOp.AND, self, x)
  def or_(self, x): return Bin(ArithOp.OR, self, x)

def symbols(syms: str) -> Tuple[sym,...]:
  return tuple([Symbol(x) for x in syms.split(',')])

class Symbol(Expr):
  __match_args__ = ('name', )
  def __init__(self, name:str, domain: Optional[Interval]=None):
    self.name: str = name
    self.domain: Optional[Interval] = domain
  def __repr__(self): return f"(Sym {self.name} {self.domain})" 
  def __str__(self): return self.name

class Lit(Expr):
  __match_args__ = ('val', )
  def __init__(self, val:int): self.val = val
  def __repr__(self): return f"(Lit {self.val})"
  def __str__(self): return f"{self.val}"

class Bin(Expr):
  __match_args__ = ('op', 'lhs', 'rhs')
  def __init__(self, op, lhs: Expr, rhs: Expr):
    self.op, self.lhs, self.rhs = op, lhs, rhs
  def precedence(self) -> int: return precedence[self.op] 
  def __repr__(self):
    return f"(Bin {self.op} {self.lhs} {self.rhs})"
  def __str__(self):
    return f"{self.lhs} {self.op.show()} {self.rhs}"

class Unary(Expr):
  __match_args__ = ('op', 'arg')
  def __init__(self, op, arg: Expr):
    self.op, self.arg = op, arg 
  def __repr__(self):
    return f"(Unary {self.op} {self.arg})"
  def __str__(self):
    return f"{self.op.show()}{self.arg}"

class IfElse(Expr):
  __match_args__ = ('cond', 'a', 'b')
  def __init__(self, cond: Expr, a: Expr, b: Expr):
    self.cond, self.a, self.b = cond, a, b
  def __repr__(self):
    return f"(IfElse {self.cond} {self.a} {self.b})"
  def __str__(self):
    return f"{self.cond} ? {self.a} : {self.b}"

def render(expr: Expr, prec:int=0) -> str:
  match expr:
    case Lit(val):
      return f"{val}" 
    case Symbol(name):
      return f"{name}" 
    case Bin(op, lhs, rhs):
      res = f"{render(lhs, expr.precedence())} {op.show()} {render(rhs, expr.precedence() + 1)}"
      return res if expr.precedence() > prec else f"({res})"
    case Unary(op, lhs):
      return f"{op.show()}{render(lhs)}"
    case IfElse(cond, a, b):
      return f"{render(cond)} ? {render(a)} : {render(b)}"
    case _:
      raise ValueError(f"Invalid expression value: {repr(expr)}")

sint: TypeAlias = Symbol|int
sym: TypeAlias = Symbol