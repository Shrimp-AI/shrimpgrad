from shrimpgrad import Tensor 

import unittest

class TestOps(unittest.TestCase):
  def test_add_1d(self):
    t1 = Tensor.arange(1)
    t2 = Tensor.arange(1)
    self.assertEqual([0], (t1+t2).data)
  
  def test_add_3d_to_1d(self):
    t1 = Tensor.ones((1000,)).reshape(10,10,10)
    t2 = Tensor.ones((1,))
    t3 = t1 + t2

    self.assertEqual(t3.shape, t1.shape)
    self.assertEqual(t3.data, [2.0]*1000)