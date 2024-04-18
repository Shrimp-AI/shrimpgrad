from shrimpgrad import Tensor, pad_left, broadcast_shape

import unittest

class TestBroadcast(unittest.TestCase):
  def test_pad_left(self):
    t1 = Tensor.arange(2*2*2*2).reshape(2,2,2,2)
    t2 = Tensor.arange(2).reshape(1,2)
    x = pad_left(t1.shape, t2.shape)
    self.assertEqual(x[0], (2,2,2,2))
    self.assertEqual(x[1], (1,1,1,2))

  def test_pad_left1(self):
    t1 = Tensor.arange(2*2).reshape(2,2)
    t2 = Tensor.arange(2).reshape(1,2)
    x = pad_left(t1.shape, t2.shape)
    self.assertEqual(x[0], (2,2))
    self.assertEqual(x[1], (1,2))
  
  def test_broadcast_shape(self):
    bs = broadcast_shape((1,1,3,4), (5,2,1,4))
    self.assertEqual(bs, (5,2,3,4))