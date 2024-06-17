import unittest
from shrimpgrad.dtype import dtypes
from shrimpgrad.engine.lower import LowIR, LowIRGraph, LowerFusedKernel
from shrimpgrad.engine.scheduler import FusedKernelBuilder, print_schedule
from shrimpgrad.runtime.ops import BinaryOps
from shrimpgrad.tensor import Tensor
import numpy as np

class TestLower(unittest.TestCase):
  def ae(self, a, b): self.assertEqual(a,b)
  def test_const(self):
    g = LowIRGraph()
    c0 = g.const(dtypes.int32, 10)
    self.ae(c0.val, 10)
    self.ae(c0.ancestors, ())
    self.ae(c0.op, LowIR.CONST)
    self.ae((), c0.ancestors)
    self.ae(c0, g.G[0])

  def test_global(self):
    g = LowIRGraph()
    g0 = g.define_global('data0', dtypes.float32, True, 0)
    self.ae(g0.ancestors, ())
    self.ae(g0.name, 'data0')
    self.ae(g0.op, LowIR.GLOBAL)
    self.ae(g0.mutable, True)
    self.ae(g0.pos, 0)

  def test_local(self):
    g = LowIRGraph()
    c0 = g.const(dtypes.float32, 10.0)
    v1 = g.local_var('idx0', dtypes.float32, c0)
    self.ae(v1.ancestors, (c0,))
    self.ae(v1.name, 'idx0')
    self.ae(v1.op, LowIR.LOCAL)

  def test_load(self):
    g = LowIRGraph()
    g0 = g.define_global('data0', dtypes.float32, True, 0)
    addr = g.address([],2, 1)
    load = g.load(g0, addr)
    self.ae(load.ancestors[0], g0)
    self.ae(load.ancestors[1], addr)

  def test_accumulate(self):
    g = LowIRGraph()
    g0 = g.define_global('data0', dtypes.float32, True, 0)
    g1 = g.define_global('data1', dtypes.float32, True, 1)
    addr1 = g.address([0], (0,), 0)
    addr2 = g.address([0], (0,), 0)
    g0_loaded = g.load(g0, addr1)
    g1_loaded = g.load(g1, addr2)
    acc = g.accumulator(BinaryOps.ADD, 'acc0', dtypes.float32, (g1_loaded, g0_loaded))
    self.ae(acc.alu, BinaryOps.ADD)
    self.ae(acc.ancestors, (g1_loaded, g0_loaded))
    self.ae(acc.op, LowIR.ACC)
    self.ae(acc.name, 'acc0')

  def test_store(self):
    g = LowIRGraph()
    g0 = g.define_global('data0', dtypes.float32, True, 0)
    addr = g.address([0],(0,),0)
    g1 = g.define_global('data1', dtypes.float32, True, 1)
    g1l = g.load(g1, addr)
    addr2 = g.address([0,],(0,),0)
    store = g.store(g0, addr2, g1l)
    self.ae(store.ancestors, (g0, addr2, g1l))

  def test_alu(self):
    g = LowIRGraph()
    g0 = g.define_global('data0', dtypes.float32, True, 0)
    addr = g.address([0],(0,),0)
    g1 = g.define_global('data1', dtypes.float32, True, 1)
    g1l = g.load(g1, addr)
    addr2 = g.address([0],(0,),0)
    g0l = g.load(g0, addr2)
    alu = g.alu(BinaryOps.ADD, dtypes.float32, *(g1l, g0l))
    self.ae(alu.alu, BinaryOps.ADD)
    self.ae(alu.ancestors, (g1l, g0l))

  def test_loop(self):
    g = LowIRGraph()
    c0 = g.const(dtypes.int32, 0)
    c1 = g.const(dtypes.int32, 10)

    loop = g.begin_loop(c0, c1)
    end_loop = g.end_loop(loop)
    self.ae(loop.ancestors, (c0, c1))
    self.ae(end_loop.ancestors, (loop, ))

  def test_lower_binary(self):
    x = Tensor.rand(2,2)
    y = Tensor.rand(2,2)
    out = x + y
    fkb = FusedKernelBuilder(out.thunk)
    schedule = fkb.schedule()
    # 2 copies and 1 add kernel
    self.assertEqual(3, len(schedule))
    print_schedule(schedule)
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    for ir_graph in ir_graphs:
      ir_graph.print()

  def test_lower_unary(self):
    x = Tensor.rand(2,2)
    out = x.log()
    fkb = FusedKernelBuilder(out.thunk)
    schedule = fkb.schedule()
    # 1 copy, 1 const, 1 log
    self.assertEqual(3, len(schedule))
    print_schedule(schedule)
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    for ir_graph in ir_graphs:
      ir_graph.print()

  def test_lower_diamond_reduce(self):
    x = Tensor.randn(10,10)
    y = Tensor.randn(10,10)
    z = Tensor.randn(10,10)
    w = Tensor.randn(10,10)

    a = x + y

    b = z * a

    c = w * a

    d = b / c

    out = d.sum(axis=0)

    fkb = FusedKernelBuilder(out.thunk)
    schedule = fkb.schedule()
    print_schedule(schedule)
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    for ir_graph in ir_graphs:
      ir_graph.print()

  def test_reduce(self):
    x = Tensor.randn(10,10,10)
    out = x.sum(axis=0)
    fkb = FusedKernelBuilder(out.thunk)
    schedule = fkb.schedule()
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    for i, ir_graph in enumerate(ir_graphs):
      print(f"GRAPH {i}")
      ir_graph.print()

  def test_lower_full_axis_reduce(self):
    x = Tensor.ones((2,))
    y = x.sum()
    fkb = FusedKernelBuilder(y.thunk)
    schedule = fkb.schedule()
    lfk = LowerFusedKernel(schedule)
    ir_graphs = lfk.lower()
    for i, ir_graph in enumerate(ir_graphs):
      print(f"GRAPH {i}")
      ir_graph.print()
    y.realize()
    print(y.data())
    np.testing.assert_array_equal(y.data(), 2.0)
