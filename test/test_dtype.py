import unittest
from shrimpgrad.dtype import type_promotion_lattice, _make_lattice_upper_bounds

class TestDtype(unittest.TestCase):
  def test_dtype_promotion(self):
    lattice = type_promotion_lattice()
    upper_bounds = _make_lattice_upper_bounds()
    print(upper_bounds)
