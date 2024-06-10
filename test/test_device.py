from shrimpgrad.device import CPU, MallocAllocator
from shrimpgrad.runtime.clang import ClangDevice
import unittest

from shrimpgrad.runtime.python import PythonAllocator, PythonCompiler, PythonDevice, PythonRenderer, PythonRuntime


class TestDevice(unittest.TestCase):
  def test_clang_device(self):
    dev = ClangDevice()
    self.assertEqual("CLANG", dev.name)
    dev0 = ClangDevice()
    self.assertEqual(dev0, dev)
    self.assertTrue(isinstance(dev.allocator(), MallocAllocator))

  def test_host_device(self):
    dev = CPU()
    self.assertEqual("CPU", dev.name)

  def test_python_device(self):
    dev = PythonDevice()
    assert "PYTHON" == dev.name
    assert isinstance(dev.allocator(), PythonAllocator)
    assert isinstance(dev.compiler(), PythonCompiler)
    assert isinstance(dev.runtime(), PythonRuntime)
    assert isinstance(dev.renderer(), PythonRenderer)