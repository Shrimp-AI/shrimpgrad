
from typing import Type
from shrimpgrad.memory.buffer import Allocator
from shrimpgrad.meta.singleton import Singleton

class Compiler:
  def compile(self): raise NotImplementedError('implement compile') 

class Runtime:
  def exec(self): raise NotImplementedError('implement exec') 


class Device(metaclass=Singleton):
  def __init__(self, allocator: Type[Allocator], compiler: Type[Compiler], runtime: Type[Runtime]) -> None:
    self._allocator, self._compiler, self._runtime = allocator, compiler, runtime


  