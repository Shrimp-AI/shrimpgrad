import unittest

from shrimpgrad.view import View, ViewTracker

class TestView(unittest.TestCase):
  def test_view(self):
    v = View(())
    self.assertTrue(v.scalar)

  def test_reshape_permute_reshape(self):
    vt = ViewTracker.from_shape((2,2))
    assert vt.shape == (2,2)
    assert vt.strides == (2,1)
    assert len(vt.views) == 1
    assert vt.contiguous
    vt = vt.reshape((1,2,2))
    assert vt.shape == (1,2,2)
    assert vt.strides == (0,2,1)
    assert len(vt.views) == 1
    assert vt.contiguous
    vt = vt.permute((0,2,1))
    assert not vt.contiguous
    assert len(vt.views) == 1
    assert vt.shape == (1,2,2)
    assert vt.strides == (0,1,2)
    vt = vt.reshape((1,1,2,2))
    assert not vt.contiguous
    assert len(vt.views) == 2
    assert vt.shape == (1,1,2,2)
    assert vt.strides == (0,0,1,2)

  def test_pad(self):
    vt = ViewTracker.from_shape((2,2))
    vt = vt.pad(((1,1),(0,0)))
    assert vt.shape == (4,2)
    assert vt.view.mask == ((1,3),(0,2))

  def test_pad2(self):
    vt = ViewTracker.from_shape((2,2,2))
    vt = vt.pad(((2,1),(4,4),(4,4)))
    assert vt.shape == (5, 10, 10)
    assert vt.views[-1].mask == ((2,4), (4,6), (4,6))
    assert vt.numel == 5*10*10
    assert vt.strides == (4, 2, 1)
    assert vt.ndim == 3
    assert len(vt.views) == 1

  def test_permute_pad(self):
    vt = ViewTracker.from_shape((2,2,2))
    vt = vt.permute((2,1,0))
    assert vt.strides == (1,2,4)
    vt = vt.pad(((1,1),(0,0),(1,1)))
    assert vt.shape == (4,2,4)
    assert vt.strides == (1,2,4)
    assert len(vt.views) == 1

  def test_shrink(self):
    view = View((2,2,2))
    view = view.shrink(((0,1), (0,1), (0,0)))

    assert view.shape == (1,1,0)
    assert view.strides == (0,0,0)

  def test_shrink1(self):
    vt = ViewTracker.from_shape((2,4,2))
    vt = vt.shrink(((0,1),(1,3), (0,2)))

    assert vt.view.shape == (1,2,2)
    assert vt.view.strides == (0,2,1)
    assert vt.view.mask == None
    assert vt.view.offset == 0

