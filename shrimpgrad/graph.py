import graphviz


def build_graph(root):
  g = graphviz.Digraph('Computational Graph', node_attr={'shape': 'record'})

  g.graph_attr['rankdir'] = 'LR'
  visited = set()

  def recur(node):
    if node in visited:
      return
    if not node:
      return
    visited.add(node)
    parents = node.prev
    if not parents:
      return
    child_label = node.label
    child_id = str(id(node))
    child_op = node._op
    op_id = child_id + node._op
    child_data = node.data
    child_grad = node.grad
    g.node(
        child_id, f'{child_label}| data {child_data:.4f} | grad {child_grad:.4f} ')
    g.node(op_id, f'{child_op}', shape='circle')
    g.edge(op_id, child_id)
    for p in parents:
      p_id = str(id(p))
      g.node(
          p_id, f'{p.label} | data {p.data:.4f} | grad {p.grad:.4f}')
      g.edge(p_id, op_id)
      recur(p)
  recur(root)
  return g
