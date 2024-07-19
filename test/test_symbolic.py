import unittest

from shrimpgrad.symbolic import ArithOp, Bin, Lit, Symbol, Interval, render, symbols, precedence
from shrimpgrad.view import ViewTracker

class TestSymbolic(unittest.TestCase):
  def test_basic_symbol(self):
    idx0 = Symbol('idx0')
    self.assertEqual('idx0', idx0.name)
    self.assertEqual(None, idx0.domain)
  
  def test_basic_interval(self):
    domain = Interval(0,20)
    assert domain.start == 0    
    assert domain.end == 20
  
  def test_symbol_with_specific_domain(self):
    view = ViewTracker.from_shape((20,21,22))
    idxs = [Symbol(f"idx{i}", Interval(0, s)) for i, s in enumerate(view.shape)]
    for i, idx in enumerate(idxs):
      assert idx.domain is not None, "domain can't be none here"
      assert idx.domain.start == 0 and idx.domain.end == view.shape[i]
      assert idx.name == f"idx{i}"
  
  def test_symbols(self):
    x,y,z,w = symbols('x,y,z,w')
    assert x.name == "x"
    assert y.name == "y"
    assert z.name == "z"
    assert w.name == "w"
  
  def test_literal_expr(self):
    lit = Lit(4) 
    assert str(lit) == "4", "literal is not correct"
  
  def test_arith_op(self):
    assert str(ArithOp.PLUS.show()) == "+"
  
  def test_binary_expr(self):
    bin = Bin(ArithOp.PLUS, Lit(4), Lit(5))
    self.assertEqual("4 + 5", str(bin))
  
  def test_variable_expr(self):
    x = Symbol("x")
    y = Symbol("y")
    vexpr = Bin(ArithOp.PLUS, x, y) 
    self.assertEqual("x + y", str(vexpr))

  def test_symbolic_add(self):
    x = Symbol("x")
    y = Symbol("y")
    z = x + y
    assert "x + y" == str(z), "wrong str" 
  
  def test_op_precedence(self):
    assert precedence[ArithOp.PLUS] == precedence[ArithOp.SUB]
    assert precedence[ArithOp.PLUS] < precedence[ArithOp.MUL]
    assert precedence[ArithOp.MOD] > precedence[ArithOp.AND]
  
  def test_unary_op(self):
    x = Symbol("x")
    expr = -x
    assert str(expr) == "-x"

  def test_render_expr(self):
    w,x,y,z = symbols("w,x,y,z") 
    expr = w + x - z * w / y % z 
    assert "(w + x) - (((z * w) / y) % z)" == render(expr)
  
  def test_render_unary_binary(self):
    x,y,z = symbols("x,y,z")
    expr = x * y + (-z)
    assert "x * y + -z" == render(expr)
  
  def test_render_ifelse(self):
    x,y,z = symbols("x,y,z")
    expr = x.ifelse(y,z)
    assert "x ? y : z", render(expr)
  
  def test_render_complex(self):
    w,x,y,z = symbols("w,x,y,z") 
    expr = (w*y).ifelse((-x)%z, (y/w).ifelse(x,y)) 
    assert "w * y ? -x % z : y / w ? x : y" == render(expr)
  
  def test_render_pad(self):
    shape = (4,4)
    stride = (2,1)
    offset = -3
    idx0, idx1 = Symbol("idx0", Interval(0, shape[0])), \
      Symbol("idx1", Interval(0, shape[1]))
    off = Lit(offset)
    strides = [Lit(st) for st in stride]
    iexpr = idx0*strides[0]+off+idx1*strides[1]
    assert "(idx0 * 2 + -3) + (idx1 * 1)", render(iexpr)
    a = idx0*(Lit(-1)) > Lit(0) 
    b = idx1*(Lit(-1)) > Lit(0)
    c = idx0 < Lit(3)
    d = idx1 < Lit(3)
    vexpr = a.and_(b.and_(c).and_(d)) 
    assert "(idx0 * -1 > 0) && (((idx1 * -1 > 0) && (idx0 < 3)) && (idx1 < 3))", render(vexpr)
    

