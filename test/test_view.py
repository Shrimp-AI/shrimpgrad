import unittest

from shrimpgrad.view import View
from shrimpgrad.runtime.clang import ClangDevice
from shrimpgrad.dtype import dtypes

class TestView(unittest.TestCase):
  def test_view(self):
    v = View(ClangDevice(), (), dtypes.float32) 
    self.assertTrue(v._scalar)