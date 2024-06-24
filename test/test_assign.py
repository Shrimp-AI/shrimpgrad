import unittest
import numpy as np
from shrimpgrad import Tensor


class TestAssign(unittest.TestCase):
  def test_unrealized_assign(self):
    x = Tensor.full((2,2), 3.0)
    y = Tensor.full((2,2), 4.0, requires_grad=False)
    x.assign(y)
    x.realize()
    np.testing.assert_array_equal(np.array([4.0]*4).reshape(2,2), x.data())

  def test_realized_assign(self):
    x = Tensor.ones((45,65))
    x.realize()
    y = Tensor.full((45,65), 4.0)
    x.assign(y)
    x.realize()
    np.testing.assert_array_equal(x.data(), np.array([4.0]*(45*65)).reshape(45,65))

  def test_opequals(self):
    x = Tensor.ones((2,2))
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
    a = Tensor.zeros((10,10))
    a.assign(Tensor.ones((10,10)))
    b = Tensor.zeros((10,10))
    a.realize()
    b.realize()
    np.testing.assert_allclose(b.data(), 0)
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
      f(x, x.full_like(x,i))
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
    a = Tensor.ones((4,)).realize()
    old_a = a
    a.assign(Tensor.full((4,), 2.))
    # NOTE: old_a is now 2, and this would match the behavior of pytorch
    new = a + old_a
    new.realize()
    np.testing.assert_allclose(new.data(), 4)

  def test_assign_diamond_cycle(self):
    a = Tensor.ones((4,)).realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.))
    # Now times_a will be 2*3
    new = a + (times_a-1)
    np.testing.assert_allclose(new.realize().data(), 7)

  def test_assign_diamond_possible(self):
    # TODO: Torch returns 4 here
    a = Tensor.ones((4,)).realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.))
    # times_a = 6, a = 2
    new = a + (times_a-1)
    np.testing.assert_allclose(new.numpy(), 7)

  def test_assign_diamond_alt(self):
    a = Tensor.ones((4,)).realize()
    a.assign(Tensor.full((4,), 2.))
    times_a = a*3
    new = a + times_a
    np.testing.assert_allclose(new.numpy(), 8)

  def test_double_assign(self):
    a = Tensor.ones((4,)).realize()
    a += 1
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