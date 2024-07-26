from __future__ import annotations
from dataclasses import dataclass
from functools import total_ordering
import os
from typing import Any, ClassVar, Dict,  NotRequired, TypedDict, Unpack
from contextlib import ContextDecorator

class KnobArgs(TypedDict):
    DEBUG: NotRequired[int] 

class Knobs(ContextDecorator):
  def __init__(self, **knobs: Unpack[KnobArgs]):
    self.pre_ctx_state = {name:k.value for name,k in Knob._knobs.items()} 
    self.knobs = knobs.items()
      
  def __enter__(self):
    for k, v in self.knobs:  Knob(k, v) # type: ignore

  def __exit__(self, *args):
    for name, value in self.pre_ctx_state.items(): Knob(name, value) 

@total_ordering
@dataclass
class Knob:
  _knobs: ClassVar[Dict[str, Knob]] = {}
  name: str
  value: int 
  def __new__(cls, name: str, value: int):
    if name in Knob._knobs: return Knob._knobs[name]
    knob = super().__new__(cls) 
    knob.name = name
    knob.value = get_knob(name, value) 
    Knob._knobs[name] = knob
    return knob
  def __eq__(self, other: Any): return self.value == int(other) 
  def __lt__(self, other: Any): return self.value < int(other)

def get_knob(name, default): return int(os.environ.get(name, default))

DEBUG = Knob("DEBUG", get_knob("DEBUG", 0))