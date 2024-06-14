from shrimpgrad import Tensor
from shrimpgrad.engine.graph import log_thunk
from shrimpgrad.util import to_nested_list
import unittest
import pytest
import torch
import numpy as np
import time

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

  def compare(self, tts, sts, rtol, atol, show=False):
    t = tts.tolist()
    s = to_nested_list(sts, None) if not sts.is_scalar() else [sts.data]
    np.testing.assert_allclose(t,s, rtol=rtol, atol=atol)

  def test_add_1d(self):
    t1 = Tensor.ones((1,))
    t2 = Tensor.ones((1,))
    t3 = t1 + t2
    t3.realize()
    self.assertEqual(2.0, t3.data()[0])

  def test_add_3d_to_1d(self):
    t1 = Tensor.ones((1000,)).reshape(10,10,10)
    t2 = Tensor.ones((1,))
    t3 = t1 + t2
    t3.realize()
    self.assertEqual(t3.shape, t1.shape)
    np.testing.assert_array_equal(t3.data(),
                                  np.array([2.0]*1000).reshape(10,10,10)) #pylint: disable=too-many-function-args

  def test_add_scalar(self):
    x = Tensor((), 2.0)
    y = Tensor((), 3.0)
    out = x*y
    out.realize()
    self.assertEqual(6.0, out.data())

  def test_lt(self):
    x = Tensor.ones((2,2))
    y = Tensor.zeros((2,2))
    z = y < x
    z.realize()
    np.testing.assert_array_equal(np.array([True]*4).reshape(2,2), z.data())

  def test_gt(self):
    x = Tensor.ones((2,2))
    y = Tensor.zeros((2,2))
    z = x > y
    z.realize()
    np.testing.assert_array_equal(np.array([True]*4).reshape(2,2), z.data())

  def test_eq(self):
    x = 1.0
    y = Tensor.ones((2,2))
    z = x == y
    z.realize()
    np.testing.assert_array_equal(np.array([True]*4).reshape(2,2), z.data())
    x = Tensor.zeros((2,2))
    z = x == y
    z.realize()
    np.testing.assert_array_equal(np.array([False]*4).reshape(2,2), z.data())

  def test_le(self):
    x = Tensor.ones((2,2))
    y = Tensor.zeros((2,2))
    z = x <= y
    z.realize()
    np.testing.assert_array_equal(z.data(), np.array([False]*4).reshape(2,2))
    z = y <= x
    z.realize()
    np.testing.assert_array_equal(z.data(), np.array([True]*4).reshape(2,2))
    y = Tensor.ones((2,2))
    z = y <= x
    z.realize()
    np.testing.assert_array_equal(z.data(), np.array([True]*4).reshape(2,2))

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_ge(self):
    x = Tensor.ones((2,2))
    y = Tensor.zeros((2,2))
    self.assertEqual((x>=y).data, [True]*4)
    self.assertEqual((y>=x).data, [False]*4)
    y = Tensor.ones((2,2))
    self.assertEqual((x>=y).data, [True]*4)

  def test_scalar_ops_with_backprop(self):
    a = Tensor((), -4.0)
    b = Tensor((), 2.0)
    c = a + b
    c.realize()
    self.assertEqual(c.data(), -2.0)

    d = a * b + (b*b*b)
    d.realize()

    self.assertEqual(d.data(), -4.0*2+2.0*2.0*2.0)

    c = c + c + 1
    c.realize()
    self.assertEqual(c.data(), -3.0)
    c = c + 1 + c + (-a)
    c.realize()
    print(c.data())
    self.assertEqual(c.data(),  -3.0+(1+-3.0+4.0))
    d = d + d * 2 + (b + a).relu()
    d.realize()
    self.assertEqual(d.data(), -4.0*2+2.0**3 + 2*(-4.0*2+2.0**3))
    d = d + 3 * d + (b - a).relu()
    d.realize()
    self.assertEqual(d.data(), 3*( -4.0*2+2.0**3 + 2*(-4.0*2+2.0**3)) + 6.0)
    e = c - d
    e.realize()
    self.assertEqual(e.data(), (-3.0+(1+-3.0+4.0)) - (3*( -4.0*2+2.0**3 + 2*(-4.0*2+2.0**3)) + 6.0))
    f = e * e
    f.realize()
    self.assertEqual(f.data(), ((-3.0+(1+-3.0+4.0)) - (3*( -4.0*2+2.0**3 + 2*(-4.0*2+2.0**3)) + 6.0))**2)
    g = f / 2.0
    g = g + 10.0 / f
    g.realize()
    self.assertAlmostEqual(g.data(), 24.70408163265306)
    # TODO: Fix backward
    # g.backward()
    # a.grad.realize()
    # self.assertEqual(138.83381924198252, a.grad.data())
    # b.grad.realize()
    # self.assertEqual(645.5772594752186, b.grad.data())

  def test_relu(self):
    t1 = Tensor((2,2), data=[-1,-1,2,2])
    t2 = t1.relu()
    t2.realize()
    print(t2.data())
    np.testing.assert_array_equal(t2.data(), np.array([0,0,2,2]).reshape(2,2))

  def test_mul_scalar_and_tensor(self):
    t1 = Tensor((2,2), data=[2,2,2,2])
    t2 = Tensor((), 4)
    t3 = t1 * t2
    t3.realize()
    np.testing.assert_array_equal(t3.data(), np.array([8]*t1.numel).reshape(2,2))

  def test_truediv_0d_0d(self):
    t1 = Tensor((), 100.0)
    t2 = Tensor((), 20.0)
    t3 = t1 / t2
    t3.realize()
    print(t3.data())
    np.testing.assert_equal(t3.data(), 5.0)

  def test_truediv_2d_3d(self):
    t1 = Tensor((2,2), [100, 200, 300, 400])
    t2 = Tensor((2,2,2), [10.0]*8)
    t3 = t1 / t2
    log_thunk(t3.thunk)
    t3.realize()
    self.assertEqual(t3.shape, (2,2,2))
    np.testing.assert_array_equal(t3.data(),
                                  np.array([10.0, 20.0,  # pylint: disable=too-many-function-args
                                            30.0, 40.0,
                                            10.0, 20.0,
                                            30.0, 40.0]).reshape(2,2,2))

  def test_sum(self):
    x = Tensor((4,4), [1]*8 + ([2]*8))
    y = x.sum(axis=0)
    self.assertEqual(y.shape, (4,))
    y = x.sum(axis=0, keepdim=True)
    self.assertEqual(y.shape, (1,4))
    y.realize()
    np.testing.assert_array_equal(y.data(), np.array([6,6,6,6]).reshape(1,4))
    y = x.sum(axis=1, keepdim=True)
    y.realize()
    self.assertEqual(y.shape, (4,1))
    np.testing.assert_array_equal(y.data(), np.array([4,4,8,8]).reshape(4,1))

  @pytest.mark.skip(reason="Failing right now figuring out")
  def test_sum_small(self):
    x = Tensor((3,2), [1]*6)
    y = x.sum(axis=0, keepdim=True)
    y.realize()
    self.assertEqual(y.shape, (1,2))
    np.testing.assert_array_equal(y.data(), np.array([2,2]).reshape(1,2)) # pylint: disable=too-many-function-args

  @pytest.mark.skip(reason="Failing right now figuring out")
  def test_sum2(self):
    x = Tensor((5,3,5,5), [1]*(5*3*5*5))
    y = x.sum(axis=0, keepdim=True)
    y.realize()
    self.assertEqual(y.shape, (1,3,5,5))
    np.testing.assert_array_equal(y.data(), np.array([5]*(3*5*5)).reshape(1,3,5,5)) # pylint: disable=too-many-function-args

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_sum3(self):
    x = Tensor((5,3,5,5), [1]*(5*3*5*5))
    y = x.sum(axis=1, keepdim=True)
    self.assertEqual(y.shape, (5,1,5,5))
    y.realize()
    np.testing.assert_array_equal(y.data(), np.array([3]*(5*5*5)).reshape(5,1,5,5)) # pylint: disable=too-many-function-args
    y.backward()
    self.assertEqual(x.grad.shape, x.shape)

  def test_transpose(self):
    y = Tensor((2,2), [4,1,
                       2,2])
    self.assertTrue(y.thunk.vt.contiguous)
    z = y.transpose()
    self.assertEqual(z.shape, (2,2))
    self.assertFalse(z.thunk.vt.contiguous)
    z.realize()
    print(z.data())

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_dot(self):
    x = Tensor((2,2), [1,0,
                       0,1])
    y = Tensor((2,2), [4,1,
                       2,2])
    z = x @ y
    self.assertEqual([4,1,2,2], z.data)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_dotND(self):
    self.helper_test_ops([(2,2),(2,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(2,2,2), (2,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(2,2,2,2,2,2), (2,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)
    self.helper_test_ops([(3,1,4,1,5,3), (3,2)], torch_op=torch.matmul, shrimp_op=Tensor.matmul)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_exp(self):
    self.helper_test_ops([(45,65)],torch.exp, Tensor.exp)
    self.helper_test_ops([()], torch.exp, Tensor.exp)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_log(self):
    self.helper_test_ops([(45,65)],torch.log, Tensor.log)
    self.helper_test_ops([()], torch.log, Tensor.log)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_mean(self):
    self.helper_test_ops([(45,65)],torch.mean, Tensor.mean)
    self.helper_test_ops([()], torch.mean, Tensor.mean)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_square(self):
    self.helper_test_ops([(45,65)],torch.square, Tensor.square)
    self.helper_test_ops([()], torch.square, Tensor.square)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_square_mean(self):
    self.helper_test_ops([(45,65)],lambda x: torch.square(x).mean(), lambda x: x.square().mean())
    self.helper_test_ops([()], lambda x: torch.square(x).mean(), lambda x: x.square().mean())

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_transpose_(self):
    self.helper_test_ops([(45,65)],lambda x: torch.transpose(x, 0, 1), Tensor.transpose)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_dot_(self):
    self.helper_test_ops([(45,65), (45,65), (45,)],lambda x, w, bias: torch.matmul(x, w.transpose(0,1)) + bias, lambda x,w,bias: x.dot(w.transpose())+bias)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_mse(self):
    out = Tensor((5,), data=[1.0,0.0,1.0,1.0,2.0])
    target = Tensor(shape=(5,), data=[0,0,0,1,2])
    sout = out.mse(target)
    sout.backward()
    sgrad = out.grad

    out = torch.tensor([1.0,0.0,1.0,1.0,2.0], requires_grad=True)
    out.retain_grad()
    target = torch.Tensor([0,0,0,1,2])
    loss = torch.nn.MSELoss()
    tout = loss(out, target)
    tout.backward()
    tgrad = out.grad

    self.compare(tout, sout, atol=1e-6, rtol=1e-3)
    self.compare(tgrad, sgrad, atol=1e-6, rtol=1e-3)

  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_sigmoid(self):
    self.helper_test_ops([(45,65)],torch.sigmoid, Tensor.sigmoid)
    self.helper_test_ops([()], torch.sigmoid, Tensor.sigmoid)


  @pytest.mark.skip(reason="Not possible without e2e realization of the graph")
  def test_binary_cross_entropy(self):
    self.helper_test_ops([(32,10), (32,10)],
                        lambda x, y: torch.nn.functional.binary_cross_entropy(x.sigmoid(), y.clip(0,1)),
                        lambda x, y: x.sigmoid().binary_cross_entropy(y.clamp(0,1)))
    # self.helper_test_ops([(), ()], torch.sigmoid, Tensor.sigmoid)