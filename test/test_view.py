import unittest

from shrimpgrad.view import View, ViewTracker

class TestView(unittest.TestCase):
  def test_view(self):
    v = View(())
    self.assertTrue(v.scalar)

  def test_reshape(self):
    vt = ViewTracker.from_shape((10,4))
    self.assertEqual((10,4), vt.shape)
    vt = vt.reshape((40,1))
    self.assertEqual((40,1), vt.shape)

  def test_reshape_like_permute(self):
    vt = ViewTracker.from_shape((2,4))
    self.assertEqual((2,4), vt.shape)
    vt = vt.reshape((4,2))
    self.assertEqual((4,2), vt.shape)
    self.assertTrue(vt.contiguous)

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

  def test_pad2(self):
    vt = ViewTracker.from_shape((2,2,2))
    vt = vt.pad(((2,1),(4,4),(4,4)))
    assert vt.shape == (5, 10, 10)
    assert vt.numel == 5*10*10
    assert vt.strides == (100,10,1)

    assert vt.ndim == 3
    assert len(vt.views) == 1

  def test_permute_pad(self):
    vt = ViewTracker.from_shape((2,2,2))
    vt = vt.permute((2,1,0))
    assert vt.strides == (1,2,4)
    vt = vt.pad(((1,1),(0,0),(1,1)))
    assert vt.shape == (4,2,4)
    assert vt.strides == (8,4,1)
    assert len(vt.views) == 1

  def test_shrink(self):
    view = View((2,2,2))
    view = view.shrink(((0,1), (0,1), (0,0)))
    assert view.shape == (1,1,0)
    assert view.strides == (0,0,0)

  def test_shrink1(self):
    vt = ViewTracker.from_shape((2,4,2))
    vt = vt.shrink(((0,1),(1,3),(0,2)))
    assert vt.view.shape == (1,2,2)
    assert vt.view.strides == (0,2,1)

  def test_undo_pad_with_shrink_and_mask(self):
    vt = ViewTracker.from_shape((4,7,4))
    vt = vt.pad(((2,1),(0,0),(1,2)))
    self.assertEqual((7,7,7), vt.shape)
    vt = vt.shrink(((2,6), (0,7), (1,5)))
    self.assertEqual((4,7,4), vt.shape)


  def test_pad_then_shrink_a_bit(self):
    vt = ViewTracker.from_shape((4,))
    vt = vt.pad(((2,2),))
    self.assertEqual((8,), vt.shape)
    vt = vt.shrink(((1,8),))
    self.assertEqual((7,), vt.shape)


  def test_pad_then_shrink_into_outer_pad(self):
    vt = ViewTracker.from_shape((4,))
    vt = vt.pad(((2,2),))
    self.assertEqual((8,), vt.shape)
    vt = vt.shrink(((1,4),))
    self.assertEqual((3,), vt.shape)

  def test_shrink_pad_back(self):
    vt = ViewTracker.from_shape((4,))
    vt = vt.shrink(((1,3),))
    self.assertEqual((2,), vt.shape)
    vt = vt.pad(((1,1),))
    self.assertEqual((4,),vt.shape)


  def test_pad_reshape_adjusts_mask(self):
    vt = ViewTracker.from_shape((2,2))
    vt = vt.pad(((1,1),(1,1)))
    self.assertEqual((4,4), vt.shape)
    vt = vt.reshape((1,4,4))
    self.assertEqual((1,4,4), vt.shape)


  def test_pad_reshape_adjust_mask2(self):
    vt = ViewTracker.from_shape((8,2))
    vt = vt.pad(((1,1),(1,1)))
    self.assertEqual((10,4), vt.shape)
    vt = vt.reshape((40,1))
    self.assertEqual((40,1), vt.shape)