from shrimpgrad import Tensor, pad_left, broadcast_shape

import unittest
import pytest


class TestBroadcast(unittest.TestCase):
  def test_pad_left(self):
    t1 = Tensor.arange(0, 2*2*2*2).reshape(2,2,2,2)
    t2 = Tensor.arange(0,2).reshape(1,2)
    x = pad_left(t1.shape, t2.shape)
    self.assertEqual(x[0], (2,2,2,2))
    self.assertEqual(x[1], (1,1,1,2))

  def test_pad_left1(self):
    t1 = Tensor.arange(0,2*2).reshape(2,2)
    t2 = Tensor.arange(0,2).reshape(1,2)
    x = pad_left(t1.shape, t2.shape)
    self.assertEqual(x[0], (2,2))
    self.assertEqual(x[1], (1,2))
  
  def test_broadcast_shape(self):
    bs = broadcast_shape((1,1,3,4), (5,2,1,4))
    self.assertEqual(bs, (5,2,3,4))

  def test_broadcast_to(self):
    t1 = Tensor.arange(0,2*2*2*2).reshape(2,2,2,2)
    t2 = Tensor.arange(0,2).reshape(1,2)
    new_shapes = pad_left(t1.shape, t2.shape)
    self.assertEqual(new_shapes, [(2,2,2,2), (1,1,1,2)])
    bs = broadcast_shape(*new_shapes)
    self.assertEqual(bs, (2,2,2,2))
    t3 = t1.broadcast_to(bs)
    self.assertEqual(t1,t3)
    t4 = t2.broadcast_to(bs)
    self.assertEqual(t4.shape, bs)
    self.assertEqual(t4.thunk.strides, (0,0,0,1))

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_broadcasted_mul1(self):
    t1 = Tensor.arange(0,2*2*2*2).reshape(2,2,2,2)
    t2 = Tensor.arange(0,2).reshape(1,2)
    t3 = t1*t2 

    self.assertEqual([0.0,1.0,0.0,3.0,0.0,5.0,0.0,7.0,0.0,9.0], t3.data[0:10])
  
  def test_broadcast_no_padding(self):
    t1 = Tensor.arange(0, 4).reshape(2,1,2)
    t2 = Tensor.arange(0, 4).reshape(1,2,2)
    t3 = t1*t2
    self.assertEqual(t3.shape,(2,2,2))

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_broadcasted_add(self):
    t1 = Tensor.arange(0,2*2*2*2).reshape(2,2,2,2)
    t2 = Tensor.arange(0,2).reshape(1,2)
    t3 = t1+t2 

    self.assertEqual([0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0], t3.data[0:10])
  
  def test_invalid_shapes_for_broadcasting(self):
    t1 = Tensor((2,2), [2,2,2,2])
    t2 = Tensor((3,3), [1]*9)
    with self.assertRaises(AssertionError):
      _ = t1*t2