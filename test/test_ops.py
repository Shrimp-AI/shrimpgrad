from shrimpgrad import Tensor, dtypes
from shrimpgrad.util import prod, to_nested_list
import unittest
import torch
import numpy as np
import time

def gen_torch_tensors(*shapes): return [torch.arange(prod(s), dtype=torch.float32, requires_grad=True).reshape(*s)for s in shapes]
def gen_shrimp_tensors(*shapes): return [Tensor.arange(0, prod(s), dtype=dtypes.float32).reshape(*s) for s in shapes]

def prepare_tensors(shapes, low=-1.5, high=1.5):
  np.random.seed(0)
  np_data = [np.random.uniform(low=low, high=high, size=shp).astype(np.float32) for shp in shapes]
  tts = [torch.tensor(data, requires_grad=True) for data in np_data]
  sts = [Tensor(shp, data.flatten().tolist() if len(shp) else data.flatten().tolist()[0]) for shp, data in zip(shapes, np_data)]
  return tts, sts

class TestOps(unittest.TestCase):
  def helper_test_ops(self, shapes, 
                      torch_op, shrimp_op, 
                      atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                      low=-1.5, high=1.5):
    torch_ts, shrimp_ts = prepare_tensors(shapes, low, high) 
    for x in torch_ts: 
      x.retain_grad()
    torch_fwd_s = time.monotonic()
    tr = torch_op(*torch_ts)
    torch_fwd_t = time.monotonic() - torch_fwd_s

    shrimp_fwd_s = time.monotonic()
    sr = shrimp_op(*shrimp_ts) 
    shrimp_fwd_t = time.monotonic() - shrimp_fwd_s

    self.compare(tr, sr, rtol=rtol, atol=atol)

    torch_bs = time.monotonic()
    tr.backward(gradient=torch.ones_like(tr)) 
    torch_bt = time.monotonic() - torch_bs
    
    shrimp_bs = time.monotonic()
    sr.backward()
    shrimp_bt = time.monotonic() - shrimp_bs

    [self.compare(xt.grad, xs.grad, rtol=grad_rtol, atol=grad_atol) for xt, xs in zip(torch_ts, shrimp_ts)]

    print("\ntesting %40r   torch/shrimp fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % \
      (shapes, torch_fwd_t*1000, shrimp_fwd_t*1000, torch_bt*1000, shrimp_bt*1000), end="")


  def compare(self, tts, sts, rtol, atol):
    t = tts.tolist()
    s = to_nested_list(sts, None)
    np.testing.assert_allclose(t,s, rtol=rtol, atol=atol) 
  
  def test_add_1d(self):
    t1 = Tensor.arange(0,1)
    t2 = Tensor.arange(0,1)
    self.assertEqual([0], (t1+t2).data)
  
  def test_add_3d_to_1d(self):
    t1 = Tensor.ones((1000,)).reshape(10,10,10)
    t2 = Tensor.ones((1,))
    t3 = t1 + t2
    self.assertEqual(t3.shape, t1.shape)
    self.assertEqual(t3.data, [2.0]*1000)
  
  def test_add_scalar(self):
    x = Tensor((), 2.0)
    y = Tensor((), 3.0)
    self.assertEqual(6.0, (x*y).data)
  
  def test_scalar_ops_with_backprop(self):
    a = Tensor((), -4.0)
    b = Tensor((), 2.0)
    c = a + b
    self.assertEqual(c.data, -2.0)
    d = a * b + (b*b*b)
    self.assertEqual(d.data, -4.0*2+2.0*2.0*2.0)
    c += c + 1
    self.assertEqual(c.data, -3.0)
    c += 1 + c + (-a)
    self.assertEqual(c.data,  -3.0+(1+-3.0+4.0))
    d += d * 2 + (b + a).relu()
    self.assertEqual(d.data, -4.0*2+2.0**3 + 2*(-4.0*2+2.0**3))
    d += 3 * d + (b - a).relu()
    self.assertEqual(d.data, 3*( -4.0*2+2.0**3 + 2*(-4.0*2+2.0**3)) + 6.0)
    e = c - d
    self.assertEqual(e.data, (-3.0+(1+-3.0+4.0)) - (3*( -4.0*2+2.0**3 + 2*(-4.0*2+2.0**3)) + 6.0))
    f = e * e
    self.assertEqual(f.data, ((-3.0+(1+-3.0+4.0)) - (3*( -4.0*2+2.0**3 + 2*(-4.0*2+2.0**3)) + 6.0))**2)
    g = f / 2.0
    g += 10.0 / f
    self.assertAlmostEqual(g.data, 24.70408163265306)
    g.backward()
    self.assertEqual(138.83381924198252, a.grad.item())
    self.assertEqual(645.5772594752186, b.grad.item())

  def test_relu(self):
    t1 = Tensor((2,2), data=[-1,-1,2,2])
    t2 = t1.relu()
    self.assertEqual(t2.data, [0,0,2,2])
  
  def test_invalid_mul_scalar_and_tensor(self):
    t1 = Tensor((2,2), data=[2,2,2,2])
    t2 = Tensor((), 4)
    with self.assertRaises(AssertionError):
      _ = t1 * t2

  def test_truediv_0d_0d(self):
    t1 = Tensor((), 100.0)
    t2 = Tensor((), 20.0)
    t3 = t1 / t2
    self.assertEqual(t3.data, 5.0)
  
  def test_truediv_2d_3d(self):
    t1 = Tensor((2,2), [100, 200, 300, 400])
    t2 = Tensor((2,2,2), [10.0]*8)
    t3 = t1 / t2
    self.assertEqual(t3.shape, (2,2,2))
    self.assertEqual(t3.data, [10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0])

  def test_sum(self):
    x = Tensor((4,4), [1]*8 + ([2]*8))
    y = x.sum(axis=0)
    self.assertEqual(y.shape, (4,))
    y = x.sum(axis=0, keepdim=True)
    self.assertEqual(y.shape, (1,4))
    self.assertEqual(y.data, [6,6,6,6])
    y = x.sum(axis=1, keepdim=True)
    self.assertEqual(y.shape, (4,1))
    self.assertEqual(y.data, [4,4,8,8])

  def test_sum2(self):
    x = Tensor((5,3,5,5), [1]*(5*3*5*5))
    y = x.sum(axis=0, keepdim=True)
    self.assertEqual(y.shape, (1,3,5,5))
    self.assertEqual(y.data, [5]*(3*5*5))

  def test_sum3(self):
    x = Tensor((5,3,5,5), [1]*(5*3*5*5))
    y = x.sum(axis=1, keepdim=True)
    self.assertEqual(y.shape, (5,1,5,5))
    self.assertEqual(y.data, [3]*(5*5*5))
    y.backward()
    self.assertEqual(x.grad.shape, x.shape)
  
  def test_transpose(self):
    y = Tensor((2,2), [4,1,
                       2,2])
    self.assertTrue(y.contiguous)
    z = y.transpose()
    self.assertEqual(z.shape, (2,2))
    self.assertEqual('tensor([[4, 2], [1, 2]])', z.__str__())
    self.assertFalse(z.contiguous)
   
  def test_dot(self):
    x = Tensor((2,2), [1,0,
                       0,1])
    y = Tensor((2,2), [4,1,
                       2,2])
    z = x.matmul(y)
    self.assertEqual([4,1,2,2], z.data)
  
  def test_dotND(self):
    self.helper_test_ops([(2,2),(2,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(2,2,2), (2,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(2,2,2,2,2,2), (2,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(3,1,4,1,5,3), (3,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
  
  def test_dot_backward(self):
    x = Tensor((2,2), [1,2,3,4]) 
    y = Tensor((2,2), [1,2,3,4])
    z = x.matmul(y)
    z.backward()
  
  def test_exp(self):
    self.helper_test_ops([(45,65)],torch.exp, Tensor.exp)
    self.helper_test_ops([()], torch.exp, Tensor.exp)
