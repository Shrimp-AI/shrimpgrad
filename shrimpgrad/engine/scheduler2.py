

from shrimpgrad.engine.postdomtree import PostDomTree
from shrimpgrad.future import Thunk


def const_folding():
  pass


def schedule_loads():
  pass


def gen_fused_kernels():
  pass


def schedule_fused_kernels():
  pass



class KernelBuilder:
  def __init__(self, out: Thunk):
    self.dom_tree = PostDomTree(out)
    self.graph = self.dom_tree.graph
    