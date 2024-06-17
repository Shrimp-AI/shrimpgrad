from dataclasses import dataclass
from typing import Final, TypeAlias, Union

ConstType: TypeAlias = Union[float, int, bool]

@dataclass(frozen=True, order=True)
class DType:
  bytes: int
  name: str
  def __repr__(self):
    return f'shrimp.{self.name}'

class dtypes:
  int32: Final[DType] = DType(4, "int32")
  float32: Final[DType] = DType(4, "float32")
  bool: Final[DType] = DType(1, "bool")
  @staticmethod
  def from_py(x: ConstType) -> DType:
    if isinstance(x, float): return dtypes.float32
    if isinstance(x, int): return dtypes.int32
    return dtypes.bool

  @staticmethod
  def cast(dtype: DType, x: ConstType) -> ConstType:
    assert dtype in (dtypes.int32, dtypes.float32, dtypes.bool), f'invalid dtype {dtype}'
    if dtype == dtypes.float32: return float(x)
    if dtype == dtypes.int32: return int(x)
    if dtype == dtypes.bool: return bool(x)

