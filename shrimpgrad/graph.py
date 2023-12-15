"""
Module provides visualization for the computational graph
represented by a DAG of Value objects.
"""
import graphviz


def build_graph(root):
    """Builds a graphviz image of the comp DAG

    Args:
        root (Value): the DAG root

    Returns:
        Digraph: a graphviz Digraph object
    """
    g = graphviz.Digraph('Computational Graph', node_attr={'shape': 'record'})

    g.graph_attr['rankdir'] = 'LR'

    def recur(node):
        if not node:
            return
        parents = node.prev
        if not parents:
            return
        child_label = node.label
        child_op = node.get_op()
        child_data = node.data
        child_grad = node.grad
        op_label = f'{child_op}_{child_label}'
        g.node(
            child_label, f'{child_label}| data {child_data:.4f} | grad {child_grad:.4f} ')
        g.node(op_label, f'{child_op}', shape='circle')
        g.edge(op_label, child_label)
        for p in parents:
            g.node(
                p.label, f'{p.label} | data {p.data:.4f} | grad {p.grad:.4f}')
            g.edge(p.label, op_label)
            recur(p)
    recur(root)
    return g
