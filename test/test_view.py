import unittest

from shrimpgrad.view import View

class TestView(unittest.TestCase):
  def test_view(self):
    v = View(()) 
    self.assertTrue(v.scalar)