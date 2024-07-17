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

def measure_speed(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time, start_time, result

N = 16384 
class TestCreationSpeed(unittest.TestCase):

  def test_full(self):
    e, s,  x_shrimp = measure_speed(lambda : Tensor.full((10000,10000), 3.0).realize())
    gbps = x_shrimp.thunk.base.buff.nbytes*1e-9/(e-s)
    print(f"shrimp load speed {gbps:.2f} GB/s")
    self.assertGreater(gbps, 0.1)  # more than 600 GB/s

    shrimp_time = e-s

    e, s, x_torch = measure_speed(torch.full, (10000,10000), 3.0)
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
