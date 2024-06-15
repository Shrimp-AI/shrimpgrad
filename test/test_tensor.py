from shrimpgrad.tensor import Tensor
from shrimpgrad.dtype import dtypes
import unittest
import numpy as np

class TestTensor(unittest.TestCase):
  def test_full1(self):
    x = Tensor.full((), 3, dtype=dtypes.int32)
    self.assertTrue(x.is_scalar())
    x.realize()
    self.assertEqual(3, x.data())
  
  def test_full2(self):
    x = Tensor.full((2,3), fill_value=5)
    x.realize()
    np.testing.assert_array_equal(x.data(), np.array([5.0]*6).reshape(2,3))
  
  def test_eye1(self):
    x = Tensor.eye(2)
    x.realize()
    np.testing.assert_array_equal(x.data(), np.array([1.0,0.0,0.0,1.0]).reshape(2,2))
    x = Tensor.eye(1)
    x.realize()
    np.testing.assert_array_equal(x.data(), np.array([1.0]).reshape(1,1))
    np.testing.assert_array_equal(Tensor.eye(3).realize().data(), np.array([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]).reshape(3,3))