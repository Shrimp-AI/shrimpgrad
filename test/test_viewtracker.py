import unittest
from shrimpgrad.view import ViewTracker


class ViewTrackerTest(unittest.TestCase):
  def test_viewtracker1(self):
    vt = ViewTracker.from_shape((2,2))
    self.assertEqual(1, len(vt.views))
    self.assertEqual((2,2), vt.view.shape)
    self.assertEqual((2,1), vt.view.strides)

    vt2 = vt.reshape((2,2,1))
    self.assertEqual(1, len(vt2.views))
    self.assertEqual((2,2,1), vt2.view.shape)
    self.assertEqual((2,1,1), vt2.view.strides)

    vt3 = vt2.permute((2,1,0))
    self.assertEqual(1, len(vt3.views))
    self.assertFalse(vt3.view.contiguous)

    vt4 = vt3.expand((2,2,2))
    self.assertEqual(1, len(vt4.views))
    self.assertEqual((2,2,2), vt4.view.shape)

  def test_permute_reshape_expand(self):
    vt = ViewTracker.from_shape((2,2,2))
    vt2 = vt.permute((2,1,0))
    self.assertEqual(1, len(vt2.views))
    self.assertEqual((2,2,2), vt2.view.shape)
    self.assertEqual((1,2,4), vt2.view.strides)
    # 2,2,2,1,1
    # 4,2,1,1,1
    # 1,2,4,0,0

    vt3 = vt2.reshape((2,2,2,1,1))
    self.assertEqual(2, len(vt3.views))
    self.assertEqual((2,2,2,1,1), vt3.view.shape)
    self.assertEqual((1,2,4,0,0), vt3.view.strides)

    vt4 = vt3.expand((2,2,2,2,2))
    self.assertEqual(2, len(vt4.views))
    self.assertEqual((2,2,2,2,2), vt4.view.shape)
    self.assertEqual((1,2,4,0,0), vt4.view.strides)