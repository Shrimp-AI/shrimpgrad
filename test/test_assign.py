import unittest
import numpy as np
from shrimpgrad import Tensor
from shrimpgrad.engine.jit import ShrimpJit


class TestAssign(unittest.TestCase):
  def test_unrealized_assign(self):
    x = Tensor.full((2,2), 3.0).contiguous()
    y = Tensor.full((2,2), 4.0, requires_grad=False)
    x.assign(y)
    x.realize()
    np.testing.assert_array_equal(np.array([4.0]*4).reshape(2,2), x.data())

  def test_realized_assign(self):
    x = Tensor.ones((45,65)).contiguous()
    x.realize()
    assert x.thunk.vt.contiguous, "should be contiguous"
    assert x.thunk.realized, "should be realized"
    y = Tensor.full((45,65), 4.0).contiguous()
    x.assign(y)
    print(x.thunk._operands)
    x.realize()
    np.testing.assert_array_equal(x.data(), np.array([4.0]*(45*65)).reshape(45,65))

  def test_opequals(self):
    x = Tensor.ones((2,2)).contiguous()
    x += 1.0
    x -= 1.0
    x *= 1.0
    x += 1.0
    x.realize()
    print(x.data())
    np.testing.assert_array_equal(x.data(), np.full((2,2), 2.0))

  def test_scalar_assign(self):
    x = Tensor.ones(())
    x += 1
    x.realize()
    np.testing.assert_array_equal(x.data(), 2.0)

  def test_assign_zeros_good(self):
    a = Tensor.zeros((10,10)).contiguous()
    a.assign(Tensor.ones((10,10)))
    b = Tensor.zeros((10,10))
    a.realize()
    np.testing.assert_allclose(b.numpy(), 0)
    np.testing.assert_allclose(a.data(), 1)

  def test_assign_add_double(self):
    def f(x):
      x += 1
      x.realize()
    x = Tensor((1,), [0])
    f(x)
    np.testing.assert_allclose(x.data(), 1)
    x = Tensor((1,), [0])
    f(x)
    np.testing.assert_allclose(x.data(), 1)

  def test_assign_other(self):
    def f(x, a):
      x.assign(a)
      x.realize()
    x = Tensor((1,),[0])
    for i in range(1, 6):
      f(x, x.full_like(x,i).contiguous())
      np.testing.assert_allclose(x.data(), i)

  def test_assign_add_other(self):
    def f(x, a):
      x += a
      x.realize()
    x = Tensor((1,),[0])
    a = 0
    for i in range(1, 6):
      a += i
      f(x, x.full_like(x,i))
      np.testing.assert_allclose(x.data(), a)

  def test_assign_changes(self):
    a = Tensor.ones((4,)).contiguous().realize()
    old_a = a
    a.assign(Tensor.full((4,), 2.).contiguous())
    # NOTE: old_a is now 2, and this would match the behavior of pytorch
    new = a + old_a
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_cycle(self):
    a = Tensor.ones((4,)).contiguous().realize()
    times_a = a*3 # 3
    # TODO: How can we force realize in the engine?
    times_a.realize()
    a.assign(Tensor.full((4,), 2.).contiguous()) # a=2
    # Now times_a will be 2*3
    new = a + (times_a-1) # 2 + (3-1) =  4
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_possible(self):
    # TODO: Torch returns 4 here
    a = Tensor.ones((4,)).contiguous().realize()
    times_a = a*3
    times_a.realize()
    a.assign(Tensor.full((4,), 2.).contiguous())
    # times_a = 6, a = 2
    new = a + (times_a-1)
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_alt(self):
    a = Tensor.ones((4,)).realize()
    a.assign(Tensor.full((4,), 2.))
    times_a = a*3
    new = a + times_a
    np.testing.assert_allclose(new.numpy(), 8)

  def test_double_assign(self):
    a = Tensor.ones((4,)).realize()
    print(a.thunk)
    a += 1
    print(a.thunk)
    a += 1
    np.testing.assert_allclose(a.numpy(),3)

  def test_crossover_assign(self):
    a = Tensor.full((4,), 2).realize()
    b = Tensor.full((4,), 3).realize()
    a += b
    b += a
    a.realize()
    b.realize()
    np.testing.assert_allclose(a.data(), 5)
    np.testing.assert_allclose(b.data(), 8)

  def test_assign_double_diamond(self):
    # TODO: Issue 7 - Double realize causes sub graph to execute twice (fix)
    a = Tensor.full((4,), 2).realize()
    b = Tensor.full((4,), 3).realize()
    a_prev = a*4
    b_prev = b+3
    b_prev.realize()
    b += a_prev
    a += b_prev
    np.testing.assert_equal(b.numpy(), 11)
    np.testing.assert_equal(a.numpy(), 8)

  def test_assign_double_diamond_reduce(self):
    # TODO: Issue 7 - Double diamond causes certain sub-expression to be evaluated twice
    a0 = Tensor.full((16, 16), 10).realize()
    a1 = Tensor.full((16, 16), 20).realize()
    b0 = Tensor.full((16, ), 1).realize()
    b1 = Tensor.full((16, ), 2).realize()
    r0 = (a0 - b1).sum(1)
    r0.realize()
    r1 = (a1 - b0).sum(1)
    r1.realize()
    b0.assign(r0 * b0)
    b1.assign(r1 * b1)
    np.testing.assert_equal(b0.numpy(), 128)
    np.testing.assert_equal(b1.numpy(), 608)

  def test_crossunder_assign(self):
    a = Tensor.full((4,), 2).realize()
    b = Tensor.full((4,), 3).realize()
    c = a+9
    c.realize()
    # Referentially Opaque:
    # For instance, a now has two meanings in the text. a = 2 or a = 2 + 3
    # Any mention of a is context dependent. c = a + 9 means a where a = 2,
    # whereas b = b + c = b + a + 9 (also wants a = 2) but ends up being a = (2+3).
    # This happens b/c realizing a leads to the backing buffer for a to store a = 2+3 = 5.
    # When we now realize b, b = b+a+9 references the backing buffer of a which now violates
    # the context dependence of a + 9 wanting a = 2 and thus  2+9, by being a = 5 adn thus 5+9.
    # Since the a buffer is used in the assign as the target buffer, the a buffer gets updated
    # before b is realized and c=a+9 is executed. c=a+9 is not in the sub-DAG of a.realize() in
    # the joint DAG of a and b.
    a += b
    b += c
    np.testing.assert_allclose(a.numpy(), 2+3)
    np.testing.assert_allclose(b.numpy(), 3+2+9)

  def test_crossunder_assign_merge(self):
    a = Tensor.full((4,), 2).realize()
    b = Tensor.full((4,), 3).realize()
    c = a+9
    c.realize()
    a += b
    b += c
    out = a*b
    out.realize()
    np.testing.assert_allclose(a.numpy(), 2+3)
    np.testing.assert_allclose(b.numpy(), 3+2+9)

  def test_simple_assignment_multioutput(self):
    a = Tensor.randn(32, 32).realize()
    b = Tensor.full((32, ), 1.).realize()
    c = Tensor.full((32, ), 2.).realize()
    d = Tensor.full((32, ), 3.).realize()

    r = a.sum(axis=1)
    b.assign(r + b)
    c.assign(r + c)
    d.assign(r + d)

    np.testing.assert_allclose(b.numpy(), a.sum(1).numpy()+1)
    np.testing.assert_allclose(c.numpy(), a.sum(1).numpy()+2)
    np.testing.assert_allclose(d.numpy(), a.sum(1).numpy()+3)

  def test_permuted_assignment_correct(self):
    a = Tensor.arange(0,4 * 4).reshape(4, 4).realize()
    b = Tensor.arange(0,4 * 4).reshape(4, 4).realize()
    a = a.permute((1, 0))
    # a is now non-contiguous
    new_val = a + b
    # new_val is contiguous 
    a.assign(new_val)
    # TODO: Fix permuted assign, we are assigning a now contiguous tensor that's actually "permuted" into
    # a permuted tensor. It's possible we need to treat rhs as a permuted tensor so it matches up.
    # np.testing.assert_equal(a.numpy(), np.arange(4 * 4).reshape(4, 4).transpose(1, 0) + np.arange(4 * 4).reshape(4, 4))

  def test_assign_add_jit(self):
    @ShrimpJit
    def f(x):
      x += 1
      return x.realize()

    x = Tensor((1,),[0])
    for _ in range(5): f(x)
    assert x.data()[0] == 5

  def test_assign_add_jit_other(self):
    @ShrimpJit
    def f(x):
      x += 1
      return x.realize()
    x = Tensor((1,),[0])
    for _ in range(5): f(x)
    assert x.data()[0] == 5

    # TODO: Input doesn't match
    # Because you need to replace the input everywhere
    # y = Tensor((1,),[0])
    # for _ in range(4): f(y)
    # assert y.data()[0] == 4
  
  def test_assign_grad_update(self):
    p1 = Tensor.randn(2,2, requires_grad=True)
    p2 = Tensor.randn(2,2, requires_grad=True)
    @ShrimpJit
    def train():
      loss =  (p1*p2).sum()
      loss.backward()
      p1.assign(p1.detach() - p1.grad)
      p2.assign(p2.detach() - p2.grad)
      p1.realize()
      p2.realize()
      return loss.realize()

    for _ in range(4):
      train()
      


