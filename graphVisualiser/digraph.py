import sys

sys.path.append("/home/2020147555/myenv/lib/python3.8/site-packages")
sys.path.append("home/2020147555/tvm/python")
import tvm
import tvm.relay as relay
import tvm.relay.testing as testing

batch_size = 1
num_class = 1000
image_shape = (3, 28, 28)

mod, params = testing.resnet.get_workload(
    num_layers=8, batch_size=batch_size, image_shape=image_shape)

def _traverse_expr(node, node_dict):
    if node in node_dict:
        return
    node_dict[node] = len(node_dict)

dot = Digraph(format='svg')
dot.attr(rankdir='BT')
dot.attr('node', shape='box')

node_dict = {}
relay.analysis.post_order_visit(mod['main'], lambda node: _traverse_expr(node, node_dict))
for node, node_idx in node_dict.items():
    if isinstance(node, relay.expr.Var):
        print(f'node_idx: {node_idx}, Var(name={node.name_hint}, type=Tensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}])')
        dot.node(str(node_idx), f'{node.name_hint}:\nTensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]')
    elif isinstance(node, relay.expr.Call):
        args = [node_dict[arg] for arg in node.args]
        print(f'node_idx: {node_idx}, Call(op_name={node.op.name}, args={args})')
        dot.node(str(node_idx), f'Call(op={node.op.name})')
        for arg in args:
            dot.edge(str(arg), str(node_idx))
    elif isinstance(node, relay.Function):
        print(f'node_idx: {node_idx}, Function(body={node_dict[node.body]})')
        dot.node(str(node_idx), f'Function')
        dot.edge(str(node_dict[node.body]), str(node_idx))
    elif isinstance(node, relay.expr.TupleGetItem):
        print(f'node_idx: {node_idx}, TupleGetItem(tuple={node_dict[node.tuple_value]}, idx={node.index})')
        dot.node(str(node_idx), f'TupleGetItem(idx={node.index})')
        dot.edge(str(node_dict[node.tuple_value]), str(node_idx))
    else:
        raise RuntimeError(f'Unknown node type. node_idx: {node_idx}, node: {type(node)}')

print(dot.render())