import unittest

from shrimpgrad import Tensor
import shrimpgrad.nn.optim as optim
import torch
import numpy as np


class TestOptimizer(unittest.TestCase):
  def test_zero_grad(self):
    t = Tensor.ones((1,1))
    o = optim.Optimizer([t])
    assert t.grad is None
    t.grad = Tensor.ones((1,1), requires_grad=False)
    assert t.grad is not None
    o.zero_grad()
    assert t.grad is None

  def test_create_sgd(self):
    t = Tensor.ones((1,1))
    sgd = optim.SGD([t], lr=0.01, momentum=0.02, dampening=1e-6, weight_decay=1e-7, nesterov=True)
    assert sgd.params[0] == t
    assert sgd.lr == 0.01
    assert sgd.momentum == 0.02
    assert sgd.dampening == 1e-6
    assert sgd.weight_decay == 1e-7
    assert sgd.nesterov == True

  def test_sgd_step(self):
    x = Tensor.ones((2,2))
    y = Tensor.full((2,2), 3.0)
    p1 = Tensor.full((2,2), 2.0, requires_grad=True)
    p2 = Tensor.full((2,2), 3.0, requires_grad=True)
    _ = x.matmul(y).matmul(p1).relu().matmul(p2).relu().sub(Tensor.full((2,2), 17.1)).square().mean().backward()
    sgd = optim.SGD([p1,p2])
    sgd.step()

    x_ = torch.ones((2,2))
    y_ = torch.full((2,2), 3.0)
    p1_ = torch.full((2,2), 2.0, requires_grad=True)
    p2_ = torch.full((2,2), 3.0, requires_grad=True)
    z_ = x_.matmul(y_).matmul(p1_).relu().matmul(p2_).relu().sub(torch.full((2,2), 17.1)).square().mean()
    z_.backward()
    sgd_ = torch.optim.SGD([p1_, p2_])
    sgd_.step()

    np.testing.assert_allclose(p1_.detach().numpy(), p1.data())
    np.testing.assert_allclose(p2_.detach().numpy(), p2.data(), rtol=1e-5, atol=1e-5)
