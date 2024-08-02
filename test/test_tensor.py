from shrimpgrad.knobs import Knobs
from shrimpgrad.tensor import Tensor
from shrimpgrad.dtype import dtypes
import unittest
import numpy as np
from shrimpgrad.util import Timing
import torch
import time

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
  
  def test_detach(self):
    x = Tensor.eye(2)
    x0 = x.detach()

    y = Tensor.ones((2,2))

    out = x * y
    out2 = x0 * y 

    Tensor.realize(out)
    Tensor.realize(out2)

    np.testing.assert_allclose(out.data(), np.array([1,0,0,1]).reshape(2,2))
    np.testing.assert_allclose(out2.data(), np.array([1,0,0,1]).reshape(2,2))

class TestTensorManipulationRoutines(unittest.TestCase): 
  def test_flatten(self):
    x = Tensor.arange(0,100).reshape(10,10)
    np.testing.assert_allclose(x.numpy(), y:=np.arange(100).reshape(10,10))
    np.testing.assert_allclose(x.flatten().numpy(), y.flatten())

  def test_flatten_non_contiguous(self):
    x = Tensor.arange(0,100).reshape(10,10).transpose().contiguous()
    np.testing.assert_allclose(x.numpy(), y:=np.arange(100).reshape(10,10).transpose())
    np.testing.assert_allclose(x.flatten().numpy(), y.flatten())
  
  def test_tile(self):
    x = Tensor((3,),[0,1,2])
    y = x.tile((2,))
    np.testing.assert_allclose(y.numpy(), np.tile(np.array([0,1,2]),2))
  
  def test_tile_1d_to_2d(self):
    x = Tensor((3,),[0,1,2])
    y = x.tile((2,2))
    np.testing.assert_allclose(y.numpy(), np.tile(np.array([0,1,2]),(2,2)))

  def test_tile_1d_to_3d(self):
    x = Tensor((3,),[0,1,2])
    y = x.tile((2,1,2))
    np.testing.assert_allclose(y.numpy(), np.tile(np.array([0,1,2]),(2,1,2)))
  
  def test_tile_2d(self):
    x = Tensor((2,2),[0,1,2,3])
    y = x.tile((2,2))
    exp = np.array([0,1,2,3]).reshape(2,2)
    exp = np.tile(exp, (2,2))
    np.testing.assert_allclose(y.numpy(), exp)
  
  def test_tile_various(self):
    x = Tensor((2,2), [1,2,3,4])
    x_ = np.array([[1,2],[3,4]])

    exp = np.tile(x_, 2)
    act = x.tile((2,)).numpy()
    np.testing.assert_allclose(exp, act)

    exp = np.tile(x_, (2,1))
    act = x.tile((2,1)).numpy()
    np.testing.assert_allclose(exp, act)

    c = np.array([1,2,3,4])
    c_ = Tensor((4,), [1,2,3,4])
    exp = np.tile(c, (4,1))
    act = c_.tile((4,1)).numpy()
    np.testing.assert_allclose(exp, act)
  
  def test_groupby(self):
    x = Tensor.arange(0,16).reshape(4,4)
    k = Tensor((2,2), [1,2,3,4])
    y = x.groupby(k.shape)
  
  def test_groupby_4d(self):
    x = Tensor.arange(0,64).reshape(2,2,4,4)
    k = Tensor.ones((1, 2, 2, 2))

    y = x.groupby(k.shape)
    print(y.numpy())
    print(y.shape)

def measure_speed(func, *args, **kwargs):
  start_time = time.time()
  result = func(*args, **kwargs)
  end_time = time.time()
  return end_time, start_time, result

N = 16384 
class TestCreationSpeed(unittest.TestCase):
  def test_full(self):
    e, s,  x_shrimp = measure_speed(lambda : Tensor.full((10_000,10_000), 3.0).realize())
    virtual_nbytes =  10_000**2 * 4
    gbps = virtual_nbytes*1e-9/(e-s)
    print(f"shrimp load speed {gbps:.2f} GB/s")
    self.assertGreater(gbps, 600)  # more than 600 GB/s

    shrimp_time = e-s

    e, s, x_torch = measure_speed(torch.full, (10_000,10_000), 3.0)
    gbps = x_torch.nbytes*1e-9/(e-s)
    print(f"torch load speed {gbps:.2f} GB/s")
    torch_time = e-s

    speed_difference = shrimp_time - torch_time

    print(f"testing Tensor.full shrimp_time={shrimp_time*1000}ms torch_time={torch_time*1000}ms delta={speed_difference*1000}ms")

    threshold = 1.3 

    np.testing.assert_allclose(x_shrimp.numpy(), x_torch.numpy())
    assert speed_difference < threshold, f"Speed difference too high: {speed_difference} seconds"
    
  def test_const_load_throughput(self):
    t = Tensor.full((N, N), 3.0)
    print(f"buffer: {t.nbytes()*1e-9:.2f} GB")
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"):
        t = Tensor.full((N, N), 3.0)
        t.realize()

class TestPadAndShrink(unittest.TestCase):
  def test_pad(self):
    t = Tensor.full((2,2), 3.0)
    np.testing.assert_allclose(t.numpy(), 3.0)
    t = t.pad(((1, 1), (1, 1))).contiguous()
    expected = np.pad(np.full((2,2),3.0), ((1,1), (1,1)))
    np.testing.assert_allclose(t.numpy(), expected)
  
  def test_pad_add(self):
    t = Tensor.full((2,2), 3.0)
    np.testing.assert_allclose(t.numpy(), 3.0)
    t = t.pad(((1, 1), (1, 1))).contiguous()
    u = Tensor.full((4,4), 5.0)
    s = t + u

    a = np.pad(np.full((2,2), 3.0), ((1,1),(1,1)))
    b = np.full((4,4), 5.0)
    c = a + b

    np.testing.assert_allclose(s.numpy(), c)

  def test_shrink(self):
    x = Tensor.full((4,4), 3.0)
    tile = x.shrink(((0,2),(0,2)))
    np.testing.assert_allclose(tile.numpy(), np.full((2,2), 3.0))
  
  def test_shrink_1d(self):
    x = Tensor.arange(0, 10)
    assert x.shape == (10,)
    y = x.shrink(((4,9),)).contiguous()
    np.testing.assert_allclose(y.numpy(), np.arange(10)[4:9])
  
  def test_pad_shrink_back(self):
    x = Tensor.full((2,2), 3.0)
    x = x.pad(((1,1), (1,1))).contiguous()
    x = x.shrink(((1,3), (1,3))).contiguous()
    np.testing.assert_allclose(x.numpy(), np.full((2,2), 3.0))
  
  def test_pad_shrink_from_copy(self):
    x = Tensor.arange(0, 4).reshape(2,2)
    x = x.pad(((1,1),(1,1)))
    x = x.shrink(((1,3),(1,3)))
    np.testing.assert_allclose(x.numpy(), np.arange(4).reshape(2,2))
  
  def test_one_dim_pad(self):
    x = Tensor.arange(0, 4).reshape(2,2)
    x = x.pad(((1,1),(0,0))).contiguous()
    np.testing.assert_allclose(x.numpy(), np.pad(np.arange(4).reshape(2,2), ((1,1),(0,0)))) 
  
  def test_one_dim_big_pad(self):
    x = Tensor.arange(0, 4).reshape(2,2)
    x = x.pad(((100,0),(0,0))).contiguous()
    np.testing.assert_allclose(x.numpy(), np.pad(np.arange(4).reshape(2,2), ((100,0),(0,0)))) 
  
class TestConv2d(unittest.TestCase):
  def test_conv(self):
    kernel = Tensor.full((2,2), 10)
    x = Tensor.full((4,4), 1.0).contiguous()
    tile = x.shrink(((0,2),(0,2)))
    s = (kernel*tile).sum()
    np.testing.assert_allclose(s.numpy(), 40.0)
  
  def test_conv2d(self):
    # (minibatch, in_channels, iH, iW)
    x = Tensor.full((1,1,10,10), 2.0)
    y = Tensor.full((1,1,2,2), 10.0) 
    with Knobs(DEBUG=4):
      print(x.conv2d(y).numpy())
    
  def test_pool_inputs(self):
    # (4,4)
    # (3,3,2,2)
    # shape = (4,4)
    # k_ = (2,2)
    # s_, d_ = (1,1), (1,1) 
    x = Tensor.arange(0,16).reshape(4,4)
    y = x.conv2d(Tensor.ones((2,2)))
    print(y.numpy())