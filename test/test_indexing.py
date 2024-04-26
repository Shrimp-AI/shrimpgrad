import unittest
from shrimpgrad import Tensor

class TestIndexing(unittest.TestCase):
  def test_scalar_indexing(self):
    x = Tensor((), 2.0)
    self.assertEqual('tensor(2.0)', x.__str__())
  
  def test_1D_indexing(self):
    x = Tensor((3,), [1,2,3])
    self.assertEqual('tensor([1, 2, 3])', x[:].__str__())
    self.assertEqual('tensor([1])', x[0].__str__())
    self.assertEqual('tensor([2])', x[1].__str__())
    self.assertEqual('tensor([1, 2])', x[0:2].__str__())
    self.assertEqual('tensor([3])', x[-1].__str__())
    self.assertEqual('tensor([2])', x[-2].__str__())
    self.assertEqual('tensor([1])', x[-3].__str__())
    self.assertEqual('tensor([1])', x[0:-2].__str__())
  
  def test_2D_indexing(self):
    x = Tensor((2,2), [i for i in range(4)])
    #[[0,1], 
    # [2,3]]
    self.assertEqual('tensor([[0]])', x[0,0].__str__())
    self.assertEqual('tensor([[3]])', x[1,1].__str__())
    self.assertEqual('tensor([[0, 1]])', x[0,:].__str__())
    self.assertEqual('tensor([[2, 3]])', x[1,:].__str__())
    self.assertEqual('tensor([[0, 1], [2, 3]])', x[:,:].__str__())
    self.assertEqual('tensor([[1], [3]])', x[:,-1].__str__())

  def test_ND_indexing(self):
    x = Tensor((2,2,2), [i for i in range(8)])
    # [[[0,1],
    #   [2,3]],
    #  [[4,5],
    #   [6,7]]]
    self.assertEqual('tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])', x.__str__())
    self.assertEqual('tensor([[[0]]])', x[0,0,0].__str__())