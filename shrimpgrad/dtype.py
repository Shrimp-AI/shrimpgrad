from __future__ import annotations
import ctypes
from dataclasses import dataclass
from typing import Final, Union
import numpy as np

ConstType = Union[float, int, bool, complex]

@dataclass(frozen=True, order=True)
class DType:
  bytes: int
  name: str
  def __repr__(self): return f'shrimp.{self.name}'

class dtypes:
  bool_: Final[DType] = DType(1, "bool")
  int8: Final[DType] = DType(1, "int8")
  int16: Final[DType] = DType(2, "int16")
  int32: Final[DType] = DType(4, "int32")
  int64: Final[DType] = DType(8, "int64")
  uint8: Final[DType] = DType(1, "uint8")
  uint16: Final[DType] = DType(2, "uint16")
  uint32: Final[DType] = DType(4, "uint32")
  uint64: Final[DType] = DType(8, "uint64")
  bfloat16: Final[DType] = DType(2, "bfloat16")
  float16: Final[DType] = DType(2, "float16")
  float32: Final[DType] = DType(4, "float32")
  float64: Final[DType] = DType(8, "float64")
  complex64: Final[DType] = DType(8, "complex64")
  complex128: Final[DType] = DType(16, "complex128")

  @staticmethod
  def from_py(x: Union[float,int,bool]) -> DType:
    t = type(x)
    if t == float: return dtypes.float32
    if t == int: return dtypes.int32
    if t == bool: return dtypes.bool_
    if t == np.uint8: return dtypes.uint8
    raise TypeError(f"dtype {t} is not supported.")

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
  if dtype == dtypes.uint8: return np.uint8
  raise TypeError(f"dtype {dtype} is not supported.")

def to_ctype(dtype: DType):
  if dtype == dtypes.float32: return ctypes.c_float
  if dtype == dtypes.int32: return ctypes.c_int
  if dtype == dtypes.bool_: return ctypes.c_bool
  if dtype == dtypes.uint8: return ctypes.c_ubyte
  raise TypeError(f"dtype {dtype} is not supported.")

_weak_types = [int, float, complex] 
_unsigned_types = [dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]
_signed_types = [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]
_float_types = [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]
_complex_types = [dtypes.complex64, dtypes.complex128]

b1_ = dtypes.bool_
i_, f_, c_ = _weak_types
u8_, u16_, u32_, u64_ = _unsigned_types
i8_, i16_, i32_, i64_ = _signed_types
f16_, bf16_, f32_, f64_ = _float_types
c64_, c128_ = _complex_types

ShrimpType = Union[DType, ConstType]
def type_promotion_lattice():
  return {
    b1_: [i_], i_: [i8_, u8_], i8_: [i16_], i16_: [i32_], i32_: [i64_], i64_: [f_],
    u8_: [i16_, u16_], u16_: [i32_, u32_], u32_: [i64_, u64_], u64_: [f_],
    f_: [c_, f16_, bf16_], c_: [c64_], c64_: [c128_], f16_: [f32_], bf16_: [f32_],
    f32_: [f64_, c64_], f64_: [c128_], c128_: []
  }

def _make_lattice_upper_bounds() -> dict[ShrimpType, set[ShrimpType]]:
  lattice = type_promotion_lattice()
  upper_bounds = {node: {node} for node in lattice}
  for n in lattice:
    while True:
      new_upper_bounds = set().union(*(lattice[b] for b in upper_bounds[n]))
      if n in new_upper_bounds:
        raise ValueError(f"cycle detected in type promotion lattice for node {n}")
      if new_upper_bounds.issubset(upper_bounds[n]):
        break
      upper_bounds[n] |= new_upper_bounds
  return upper_bounds

def type_promotion(a: ShrimpType, b: ShrimpType) -> ShrimpType:
  if a == b: return a
  N = set([a,b])
  UB = _make_lattice_upper_bounds()
  CUB = set.intersection(*(UB[n] for n in N)) 
  LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
  if len(LUB) == 0:
    raise TypeError(f"No common type found for {a} and {b}")
  return LUB.pop()