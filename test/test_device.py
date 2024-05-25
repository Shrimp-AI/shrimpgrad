from shrimpgrad.device import ClangDevice, CPU, MallocAllocator
import unittest


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

