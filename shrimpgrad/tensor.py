from typing import Iterable, Optional, Self, TypeAlias, Union
from functools import reduce
import operator
from pprint import pformat
import ctypes
from ctypes.util import find_library

libcblas = ctypes.CDLL(find_library('cblas'))
Num: TypeAlias = Union[float, int, complex]
DType: TypeAlias = Union[float, int]
Shape: TypeAlias = tuple[int]

def prod(xs: Iterable[int|float]) -> Union[float, int]:
  return reduce(operator.mul, xs, 1)

class Tensor:
  def __init__(self, shape: Shape, data: Union[Iterable[Num], Num]) -> Self:
    self.shape, self.data, self.size, self.strides = shape, data, prod(shape), [] 
    # Scalar value
    if not len(self.shape):
      # Ensure it's a number not dimensional data
      assert isinstance(data, Num)
      self.data = data
      self.base_view = data
      return
    self.__calc_strides()
    self.base_view = self.__build_view(None)

  def __calc_strides(self) -> None:
    self.strides.clear()
    out: int = 1
    for dim in self.shape:
      out *= dim
      self.strides.append(self.size // out)

  def __build_view(self, key: Optional[slice|int]) -> Iterable:
    if not key:
      key = []
      for dim in self.shape:
        s = slice(0, dim, 1)
        key.append(s)
    if isinstance(key, int) or isinstance(key, slice):
      key = (key,)
    if len(key) > len(self.shape):
      raise IndexError('index out of bounds.')
    extra_dim = 0
    if len(key) < len(self.shape):
      extra_dim = len(self.shape) - len(key)
    loops = []
    for i, k in enumerate(key):
      if isinstance(k, int):
        if k >= self.shape[i]: 
          raise IndexError(f'index out of bounds: {self.shape[i]} <= {k}')
        start = k
        if start < 0:
          start = self.shape[i] + start
        start, end = k, k + 1

        loops.append((start,end, 1))
      elif isinstance(k, slice):
        start, end, step = k.indices(self.shape[i])
        if start < 0:
          start = self.shape[i] + start 
        if end < 0:
          end = self.shape[i] + end
        if start >= end:
          return [] 
        loops.append((start, end, step))
    if extra_dim:
      start = len(key)
      for ed in range(start, start+extra_dim):
        loops.append((0, self.shape[ed], 1))

    def build(dim:int, offset:int, loops:Iterable[tuple], tensor:Iterable[Num]) -> Iterable[Num]:
      if not loops:
        return tensor
      s, e, step = loops[0]
      for i in range(s, e, step):
        if len(loops) == 1:
          # add elements
          offset += i*step
          tensor.append(self.data[offset])
          offset -= i*step
        else:
          # add dimension
          new_dim = []
          tensor.append(new_dim)
          offset += i*self.strides[dim]*step
          build(dim+1, offset, loops[1:], new_dim)
          offset -= i*self.strides[dim]*step
      return tensor 
    return build(0, 0, loops, []) 
  
  def item(self) -> Num:
    if len(self.shape):
      raise RuntimeError(f'a Tensor with {self.size} elements cannot be converted to Scalar')
    return self.data

  def __getitem__(self, key) -> Self:
    if not len(self.shape):
      raise IndexError('invalid index of a 0-dim tensor. Use `tensor.item()`')
    new_view = self.__build_view(key)
    new_tensor = Tensor(self.shape, self.data)
    new_tensor.base_view = new_view
    return new_tensor

  def matmul(self, other: Self) -> Self:
    # 1D x 1D (dot product) 
    if len(self.shape) == 1 and len(other.shape) == 1:
      if self.shape[0] != other.shape[0]:
        raise RuntimeError(f'inconsistent tensor size, expected tensor [{self.shape[0]}] and src [{other.shape[0]}] to have the same number of elements, but got {self.shape[0]} and {other.shape[0]} elements respectively')
      dot = sum(map(lambda x: x[0]*x[1], zip(self.data, other.data)))
      return Tensor((), dot)

    if len(self.shape) == 2 and len(other.shape) == 2:
      if self.shape[1] != other.shape[0]:
        raise RuntimeError('mat1 and mat2 shapes cannot be multiplied ({self.shape[0]}x{self.shape[1]} and {other.shape[0]}x{other.shape[1]})')
      result = cblas_matmul(self, other)
      return Tensor((self.shape[0], self.shape[1]), [x for x in result])

  def reshape(self, *args) -> Self: 
    new_size = prod(args)
    if new_size != self.size:
      raise RuntimeError('shape \'{args}\' is invalid for input of size {self.size}')
    self.shape = tuple(args)
    self.size = new_size
    self.__calc_strides()
    self.base_view = self.__build_view(None)
    return self

  def __repr__(self):
    return f'tensor({pformat(self.base_view, width=40)})'

def zeros(shape: Shape) -> Tensor:
  return Tensor(shape, [0]*prod(shape))

def arange(x: int) -> Tensor:
  return Tensor((x,), [float(i) for i in range(x)]) 

# Define the types for the function arguments and return value
libcblas.cblas_sgemm.restype = None
libcblas.cblas_sgemm.argtypes = [
    ctypes.c_int,  # order
    ctypes.c_int,  # transa
    ctypes.c_int,  # transb
    ctypes.c_int,  # m
    ctypes.c_int,  # n
    ctypes.c_int,  # k
    ctypes.c_float,  # alpha
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.c_int,  # lda
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.c_int,  # ldb
    ctypes.c_float,  # beta
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,  # ldc
]

def cblas_matmul(a, b):
  # Allocate memory for result matrix
  m, k  = a.shape
  k, n = b.shape
  result = (ctypes.c_float * (m * n))()
  # Call cblas_sgemm function
  # If you are using row-major representation then the 
  # number of "columns" will be leading dimension and
  # vice versa in column-major representation number of "rows".
  libcblas.cblas_sgemm(
      ctypes.c_int(101),  # CblasRowMajor
      ctypes.c_int(111),  # CblasNoTrans
      ctypes.c_int(111),  # CblasNoTrans
      ctypes.c_int(m),
      ctypes.c_int(n),
      ctypes.c_int(k),
      ctypes.c_float(1.0),
      (ctypes.c_float * a.size)(*a.data),
      ctypes.c_int(k),
      (ctypes.c_float * b.size)(*b.data),
      ctypes.c_int(n),
      ctypes.c_float(0.0),
      result,
      ctypes.c_int(n)
  )
  return result