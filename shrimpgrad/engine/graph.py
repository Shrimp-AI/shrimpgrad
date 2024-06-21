import os, atexit, functools
from collections import defaultdict
from typing import Any, DefaultDict
from shrimpgrad.runtime.ops import UnaryOps, BinaryOps, ReduceOps, LoadOps, BufferOps, TernaryOps
from shrimpgrad.future import Thunk

try: import networkx as nx
except ImportError: pass

# **** debugging and graphing ****
# From tinygrad with modifications

def save_graph(G, fn, opt=""):
  print("saving", G, f"to {fn}.svg")
  nx.drawing.nx_pydot.write_dot(G, f'{fn}.dot')
  os.system(f'dot {opt} -Tsvg {fn}.dot -o {fn}.svg')

G:Any = None
def init_graph():
  global G
  if G is not None: return
  G = nx.DiGraph()
  atexit.register(functools.partial(save_graph, G, '/tmp/net')) # -Gnslimit=100 can make it finish, but you won't like results

counts: DefaultDict[type, int] = defaultdict(int)
def nm(x):
  if not hasattr(x, 'node_id'):
    setattr(x, 'node_id', counts[type(x)])
    counts[type(x)] += 1
  return x.node_id

def realized_thunk(thunk:'Thunk', num):
  init_graph()
  G.nodes[nm(thunk)]['style'] = '"filled,bold"'
  G.nodes[nm(thunk)]['fillcolor'] = G.nodes[nm(thunk)]['fillcolor'][:-2]
  G.nodes[nm(thunk)]['label'] = '"' + G.nodes[nm(thunk)]["label"].replace('"', '') + f'\nK:{num}"'

top_colors = {LoadOps: '#FFFFa0', UnaryOps: "#c0c0c0", ReduceOps: "#FFA0A0", BinaryOps: "#c0c0c0",
              TernaryOps: "#c0c0c0", BufferOps: '#a0a0ff'}
def log_thunk(thunk:'Thunk', scheduled=False):
  init_graph()
  if thunk.base.realized is None and thunk.base._op is LoadOps.CONST: return
  if thunk.base != thunk:
    # movement op
    label = f"{thunk.shape}\n{thunk.strides}"
    G.add_node(nm(thunk), style='"filled,dashed"', fillcolor="#80ff8080", color="black", label=label)
    G.add_edge(nm(thunk.base), nm(thunk), color='#00000060')
    thunk = thunk.base
  # if thunk.realized is None:
  label_append = []
  label = 'EMPTY LOADS'
  if hasattr(thunk, '_operands'):
    for idx,x in enumerate(thunk._operands):
      if nm(x) not in G.nodes: log_thunk(x)
      if x.base.realized is None and x.base._op is LoadOps.CONST:
        label_append.append(f"\nCONST{idx} {x.base.arg}")
      else:
        G.add_edge(nm(x), nm(thunk), color='#a0a0a0')
    label = '"' + \
      (str(set(x.shape for x in thunk._operands))+"\n"+str(thunk.shape) if thunk._op in ReduceOps else str(thunk.shape)) + \
      (f"\n{thunk.dtype.name} id={id(thunk)}" if thunk.dtype.name != "float" else "")+f"\n{thunk._op}"+(f"\n{thunk.arg}" if thunk._op in {LoadOps.CONST, UnaryOps.CAST} else "") + \
      (f"\n{thunk.device}") + ''.join(label_append) + '"'
  G.add_node(nm(thunk), style='"filled,dashed"', fillcolor=[v for k,v in top_colors.items() if thunk._op in k][0] + "80", color="black", label=label)
  if scheduled: G.nodes[nm(thunk)]['shape'] = 'box'
  # else:
  #   if nm(thunk) not in G.nodes:
  #     # realized but unseen?
  #     G.add_node(nm(thunk), label=f'"{str(thunk.base.realized)[5:-1].replace(" ", chr(10))}\nb:{nm(thunk.realized)}"', style='filled', fillcolor="orange")
