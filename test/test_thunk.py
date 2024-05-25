import unittest
from shrimpgrad import Tensor
from shrimpgrad.device import CPU
from shrimpgrad.runtime.ops import LoadOps

class TestThunk(unittest.TestCase):
  def test_load_empty(self):
    x = Tensor.full((2,2), 2.0)
    self.assertEqual(x.thunk._op, LoadOps.COPY)
    src = x.thunk._operands[0]
    # Assert src is an empty load, into a realized buffer, on CPU
    self.assertEqual(src._op, LoadOps.EMPTY)
    self.assertEqual(src.realized, src.buff)
    self.assertEqual(src.device, CPU())
  
  def test_load_empty_reshape(self):
    x = Tensor.full((2,2), 2.0).reshape(*(1,2,2,))
    self.assertEqual(x.thunk._op, None)
    # After a reshape we know x.thunk is a view so get the base 
    src = x.thunk.base._operands[0]
    # Assert src is an empty load, into a realized buffer, on CPU
    self.assertEqual(src._op, LoadOps.EMPTY)
    self.assertEqual(src.realized, src.buff)
    self.assertEqual(src.device, CPU())

  def test_load_empty_double_reshape(self):
    x = Tensor.full((2,2), 2.0).reshape(*(1,2,2,)).reshape(*(2,2))
    self.assertEqual(x.thunk._op, None)
    # After two reshapes we know x.thunk is a view so get the base 
    src = x.thunk.base._operands[0]
    # Assert src is an empty load, into a realized buffer, on CPU
    self.assertEqual(src._op, LoadOps.EMPTY)
    self.assertEqual(src.realized, src.buff)
    self.assertEqual(src.device, CPU())

