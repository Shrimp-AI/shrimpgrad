import unittest

from shrimpgrad.tensor import Tensor


class TestFuseOps(unittest.TestCase):
  def test_basic_elementiwse_fuse(self):
    _ = Tensor.randn(10,10)
    # # LoadOps.EMPTY -> LoadOps.COPY
    # y = Tensor.randn(10,10)