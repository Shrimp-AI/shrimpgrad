import unittest
from shrimpgrad import Tensor
from shrimpgrad.util import deepwalk

class TestUtil(unittest.TestCase):
  def test_deepwalk(self):
    x = Tensor.ones((2,2))
    y = Tensor.ones((2,2))
    out = x / y
    for i, t in enumerate(deepwalk(out)): 
      if i == 0: assert t == x
      if i == 1: assert t == y
      if i == 2: assert t == out