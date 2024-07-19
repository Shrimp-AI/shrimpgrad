import unittest

from shrimpgrad.symbolic import ArithOp, Bin, Lit, Symbol, Interval
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
  
  def test_literal_expr(self):
    lit = Lit(4) 
    assert str(lit) == "4", "literal is not correct"
  
  def test_arith_op(self):
    assert ArithOp.PLUS.show() == "+"
  
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