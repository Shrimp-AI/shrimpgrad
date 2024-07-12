import unittest

from shrimpgrad.view import View, ViewTracker

class TestView(unittest.TestCase):
  def test_view(self):
    v = View(())
    self.assertTrue(v.scalar)

  def test_reshape_permute_reshape_expand(self):
    vt = ViewTracker.from_shape((2,2))
    assert vt.shape == (2,2)
    assert vt.strides == (2,1)
    assert len(vt.views) == 1
    assert vt.contiguous
    vt = vt.reshape((1,2,2))
    assert vt.shape == (1,2,2)
    assert vt.strides == (4,2,1)
    assert len(vt.views) == 1
    assert vt.contiguous
    vt = vt.permute((0,2,1))
    assert not vt.contiguous
    assert len(vt.views) == 1
    assert vt.shape == (1,2,2)
    assert vt.strides == (4,1,2)
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