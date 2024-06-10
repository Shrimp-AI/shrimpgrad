from shrimpgrad.tensor import Tensor
from shrimpgrad.dtype import dtypes
import unittest
import pytest

class TestTensor(unittest.TestCase):
  @pytest.mark.skip(reason="Not possible without e2e realization of the graph") 
  def test_full1(self):
    x = Tensor.full((), 3, dtype=dtypes.int32)
    self.assertTrue(x.is_scalar())
    self.assertEqual(3, x.data)
  
  @pytest.mark.skip(reason="Not possible without e2e realization of the graph") 
  def test_full2(self):
    x = Tensor.full((2,3), fill_value=5)
    self.assertEqual(x.data, [5.0]*6)
  
  @pytest.mark.skip(reason="Not possible without e2e realization of the graph") 
  def test_eye1(self):
    x = Tensor.eye(2)
    self.assertEqual(x.data, [1.0,0.0,0.0,1.0])
    x = Tensor.eye(1)
    self.assertEqual(x.data, [1.0])
    self.assertEqual(Tensor.eye(3).data, [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])