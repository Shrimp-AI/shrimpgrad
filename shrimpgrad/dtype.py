import ctypes
from dataclasses import dataclass
from typing import Final, Union
import numpy as np

ConstType = Union[float, int, bool]

@dataclass(frozen=True, order=True)
class DType:
  bytes: int
  name: str
  def __repr__(self):
    return f'shrimp.{self.name}'

class dtypes:
  int32: Final[DType] = DType(4, "int32")
  float32: Final[DType] = DType(4, "float32")
  bool_: Final[DType] = DType(1, "bool")
  @staticmethod
  def from_py(x: Union[float,int,bool]) -> DType:
    t = type(x)
    if t == float: return dtypes.float32
    if t == int: return dtypes.int32
    return dtypes.bool_

  @staticmethod
  def cast(dtype: DType, x: ConstType) -> ConstType:
    assert dtype in (dtypes.int32, dtypes.float32, dtypes.bool_), f'invalid dtype {dtype}'
    if dtype == dtypes.float32: return float(x)
    if dtype == dtypes.int32: return int(x)
    if dtype == dtypes.bool_: return bool(x)
    raise TypeError(f"dtype {dtype} is invalid.")
  

def to_numpy(dtype: DType) :
  if dtype == dtypes.float32: return np.float32
  if dtype == dtypes.int32: return np.int32
  if dtype == dtypes.bool_: return np.bool_
  raise TypeError(f"dtype {dtype} is not supported.")

def to_ctype(dtype: DType):
  if dtype == dtypes.float32: return ctypes.c_float
  if dtype == dtypes.int32: return ctypes.c_int
  if dtype == dtypes.bool_: return ctypes.c_bool
  raise TypeError(f"dtype {dtype} is not supported.")