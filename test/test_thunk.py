import unittest
from shrimpgrad import Tensor
from shrimpgrad.device import CPU
from shrimpgrad.dtype import dtypes
from shrimpgrad.future import Thunk, create_thunk
from shrimpgrad.runtime.clang import ClangDevice
from shrimpgrad.runtime.ops import LoadOps
from shrimpgrad.view import ViewTracker

class TestThunk(unittest.TestCase):
  def test_load_empty(self):
    x = Tensor((2,2), [2.0]*4)
    self.assertEqual(x.thunk._op, LoadOps.COPY)
    src = x.thunk._operands[0]
    # Assert src is an empty load, into a realized buffer, on CPU
    self.assertEqual(src._op, LoadOps.EMPTY)
    self.assertEqual(src.realized, src.buff)
    self.assertEqual(src.device, CPU())
  
  def test_load_const_reshape(self):
    x = Tensor.full((2,2), 2.0).reshape(*(1,2,2,))
    self.assertEqual(x.thunk._op, None)
    src = x.thunk.base
    self.assertEqual(src._op, LoadOps.CONST)
    self.assertTrue(src.realized is not None)
    self.assertEqual(src.device, ClangDevice())

  def test_load_empty_double_reshape(self):
    x = Tensor((2,2), [2.0]*4).reshape(*(1,2,2,)).reshape(*(2,2))
    self.assertEqual(x.thunk._op, None)
    # After two reshapes we know x.thunk is a view so get the base 
    src = x.thunk.base._operands[0]
    # Assert src is an empty load, into a realized buffer, on CPU
    self.assertEqual(src._op, LoadOps.EMPTY)
    self.assertEqual(src.realized, src.buff)
    self.assertEqual(src.device, CPU())

  def test_pad(self):
    t = create_thunk(ClangDevice(), dtypes.float32, ViewTracker.from_shape((2,2,2)), (), LoadOps.EMPTY) 
    t1 = t.pad(((1,1),(1,1),(1,1)), 0.0)
    self.assertEqual((4,4,4), t1.shape)
    self.assertEqual(0.0, t1.arg)
  
class TestLoads(unittest.TestCase):
  def test_load_const_function_nd(self):
    t = Thunk.load_const(3.0, (2,2,2), dtypes.float32, ClangDevice())
    self.assertEqual(LoadOps.CONST, t.base._op)
    self.assertEqual((), t._operands)
    self.assertEqual((2,2,2), t.shape)
    self.assertEqual(3.0,t.base.arg )
    # Load const only store one value
    self.assertEqual(4, t.base.buff.nbytes)
    self.assertEqual(True, t.base.buff.allocated)
    self.assertEqual(ClangDevice(), t.device)
    
  def test_load_const_function_scalar(self):
    t = Thunk.load_const(3.0, (), dtypes.float32, ClangDevice())
    self.assertEqual(LoadOps.CONST, t.base._op)
    self.assertEqual((), t._operands)
    self.assertEqual((), t.shape)
    self.assertEqual(3.0,t.base.arg )
    self.assertEqual(4, t.base.buff.nbytes)
    self.assertEqual(True, t.base.buff.allocated)
    self.assertEqual(ClangDevice(), t.device)

  def test_load_const(self):
    x = Tensor.full((2,2,2), 3.0)
    self.assertEqual(LoadOps.CONST, x.thunk.base._op)
    self.assertEqual(3.0, x.thunk.base.arg)