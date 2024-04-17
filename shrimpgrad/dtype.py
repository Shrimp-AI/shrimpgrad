from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True, order=True)
class DType:
  itemsize: int
  name: str
  def __repr__(self):
    return f'shrimp.{self.name}'
  
class dtypes:
  int32: Final[DType] = DType(4, "int32")
  float32: Final[DType] = DType(4, "float32")