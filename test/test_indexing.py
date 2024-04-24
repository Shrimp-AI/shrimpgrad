import unittest
from shrimpgrad import Tensor

class TestIndexing(unittest.TestCase):
  def test_scalar_indexing(self):
    x = Tensor((), 2.0)
    self.assertEqual('tensor(2.0)', x.__str__())