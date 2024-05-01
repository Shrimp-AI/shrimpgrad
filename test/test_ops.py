from shrimpgrad import Tensor, dtypes
from shrimpgrad.util import prod, to_nested_list
import unittest
import torch

def gen_torch_tensors(*shapes): return [torch.arange(prod(s), dtype=torch.float32, requires_grad=True).reshape(*s)for s in shapes]
def gen_shrimp_tensors(*shapes): return [Tensor.arange(0, prod(s), dtype=dtypes.float32).reshape(*s) for s in shapes]

class TestOps(unittest.TestCase):
  def helper_test_ops(self, x_shapes, y_shapes, torch_op, shrimp_op):
    x_torch_ts = gen_torch_tensors(*x_shapes)
    x_shrimp_ts = gen_shrimp_tensors(*x_shapes) 
    if y_shapes:
      y_torch_ts = gen_torch_tensors(*y_shapes) 
      y_shrimp_ts = gen_shrimp_tensors(*y_shapes) 
    if y_shapes:    
      for x,y in zip(x_torch_ts, y_torch_ts):
        x.retain_grad()
        y.retain_grad()
      tr = [torch_op(x,y) for x,y in zip(x_torch_ts, y_torch_ts)]
      sr = [shrimp_op(x,y) for x,y in zip(x_shrimp_ts, y_shrimp_ts)]
      [self.compare(tt, st) for tt, st in zip(tr, sr)] 
      [t.backward(gradient=torch.ones_like(t)) for t in tr]
      [s.backward() for s in sr]
      [self.compare(xt.grad, xs.grad) for xt, xs in zip(x_torch_ts, x_shrimp_ts)]
      [self.compare(yt.grad, ys.grad) for yt, ys in zip(y_torch_ts, y_shrimp_ts)]

  def compare(self, tts, sts):
    t = tts.tolist()
    s = to_nested_list(sts, None)
    self.assertEqual(t,s)

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
    z = y.transpose()
    self.assertEqual(z.shape, (2,2))
    self.assertEqual('tensor([[4, 2], [1, 2]])', z.__str__())
   
  def test_dot(self):
    x = Tensor((2,2), [1,0,
                       0,1])
    y = Tensor((2,2), [4,1,
                       2,2])
    z = x.matmul(y)
    self.assertEqual([4,1,2,2], z.data)
  
  def test_dotND(self):
    self.helper_test_ops([(2,2),(3,2)], [(2,2) ,(2,4)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(2,2,2)], [(2,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(2,2,2,2,2,2)], [(2,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(3,1,4,1,5,3)], [(3,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
  
  def test_dot_backward(self):
    x = Tensor((2,2), [1,2,3,4]) 
    y = Tensor((2,2), [1,2,3,4])
    z = x.matmul(y)
    print("Start backward")
    z.backward()
    print(f'x.grad={x.grad}')
    print(f'y.grad={y.grad}')
    print(f'z.grad={z.grad}')