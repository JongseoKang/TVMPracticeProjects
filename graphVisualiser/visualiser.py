import sys

sys.path.append("/home/2020147555/myenv/lib/python3.8/site-packages")
from graphviz import Digraph

sys.path.append("home/2020147555/tvm/python")
import tvm
import tvm.relay as relay
import tvm.relay.testing as testing

from tvm.ir import Op
from tvm.relay.function import Function, FunctionWithFields
from tvm.relay.expr import Call, Let, Var, GlobalVar
from tvm.relay.expr import If, Tuple, TupleGetItem, Constant
from tvm.relay.expr import RefCreate, RefRead, RefWrite
from tvm.relay.adt import Constructor, Match, Clause
from tvm.relay.expr_functor import ExprVisitor

class MyVisitor(ExprVisitor):
    """
    A visitor over Expr.

    The default behavior recursively traverses the AST.
    """
    def __init__(self, retIdx):
        self.nodeIdx = retIdx
        self.memo_map = {}

    def visit_tuple(self, tup):
        self.nodeIdx[tup] = len(self.nodeIdx)
        for x in tup.fields:
            self.visit(x)

    def visit_call(self, call):
        self.nodeIdx[call] = len(self.nodeIdx)
        self.visit(call.op)
        for a in call.args:
            self.visit(a)

    def visit_var(self, var):
        self.nodeIdx[var] = len(self.nodeIdx)
        pass

    def visit_let(self, let):
        self.nodeIdx[let] = len(self.nodeIdx)
        self.visit(let.var)
        self.visit(let.value)
        self.visit(let.body)

    def visit_function(self, fn):
        self.nodeIdx[fn] = len(self.nodeIdx)
        for x in fn.params:
            self.visit(x)
        self.visit(fn.body)

    def visit_if(self, i):
        self.nodeIdx[i] = len(self.nodeIdx)
        self.visit(i.cond)
        self.visit(i.true_branch)
        self.visit(i.false_branch)

    def visit_global_var(self, gv):
        self.nodeIdx[gv] = len(self.nodeIdx)
        pass

    def visit_constructor(self, c):
        self.nodeIdx[c] = len(self.nodeIdx)
        pass

    def visit_op(self, op):
        self.nodeIdx[op] = len(self.nodeIdx)
        pass

    def visit_constant(self, const):
        self.nodeIdx[const] = len(self.nodeIdx)
        pass

    def visit_ref_create(self, r):
        self.nodeIdx[r] = len(self.nodeIdx)
        self.visit(r.value)

    def visit_ref_read(self, r):
        self.nodeIdx[r] = len(self.nodeIdx)
        self.visit(r.ref)

    def visit_ref_write(self, r):
        self.nodeIdx[r] = len(self.nodeIdx)
        self.visit(r.ref)
        self.visit(r.value)

    def visit_tuple_getitem(self, t):
        self.nodeIdx[t] = len(self.nodeIdx)
        self.visit(t.tuple_value)

    def visit_match(self, m):
        self.nodeIdx[m] = len(self.nodeIdx)
        self.visit(m.data)
        for c in m.clauses:
            self.visit(c.rhs)

class Visualiser():
    def __init__(self, mod, outFileName, outFileFormat):
        self.outFileName = outFileName
        self.outFileFormat = outFileFormat
        self.mod = mod
        

    def run(self):
        nodeDict = {} 
        mod = self.mod
        dot = Digraph(format=self.outFileFormat)
        dot.attr('node', shape='box')
        dot.attr(rankdir='TB')
        visitor = MyVisitor(nodeDict)
        visitor.visit(mod["main"])
        for node, nodeIdx in nodeDict.items():
            if isinstance(node, Function):
                funcName = ''
                for pair in resnet.functions_items():
                    if pair[1] == node:
                        funcName = pair[0]
                        break
                dot.node(str(nodeIdx), f'Function: {funcName}')
                dot.edge(str(nodeDict[node.body]), str(nodeIdx))
                for param in node.params:
                    dot.edge(str(nodeIdx), str(nodeDict[param]))

            elif isinstance(node, Call):
                args = [nodeDict[arg] for arg in node.args]
                dot.node(str(nodeIdx), f'Call(op={node.op.name})', fillcolor='green', style='filled')
                for arg in args:
                    dot.edge(str(arg), str(nodeIdx))

            elif isinstance(node, Let):
                print("Let node exists")

            elif isinstance(node, Var):
                dot.node(str(nodeIdx), f'{node.name_hint}:\nshape=[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]', fillcolor='yellow', style='filled')
                
            elif isinstance(node, GlobalVar):
                # maybe for tensors not function names
                dot.node(str(nodeIdx), f'{node.name_hint}:\nshape=[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]')

            elif isinstance(node, If):
                dot.node(str(nodeIdx), f'If')
                dot.edge(str(nodeDict[node.cond]), str(nodeIdx))
                dot.edge(str(nodeIdx), str(nodeDict[node.true_branch]), "TRUE")
                dot.edge(str(nodeIdx), str(nodeDict[node.false_branch]), "FALSE")

            elif isinstance(node, Tuple):
                dot.node(str(nodeIdx), f'Tuple')
                for field in node.fields:
                    dot.edge(str(nodeDict[field], str(nodeIdx)))

            elif isinstance(node, TupleGetItem):
                dot.node(str(nodeIdx), f'TupleGetItem(idx={node.index})', fillcolor='red', style='filled')
                dot.edge(str(nodeDict[node.tuple_value]), str(nodeIdx))

            elif isinstance(node, Constant):
                dot.node(str(nodeIdx), f'Const: shape=[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]', fillcolor='blue', style='filled')

            elif isinstance(node, Op):
                print("Op node exists")
            #     dot.node(str(nodeIdx), f'{node.name}')
            elif isinstance(node, RefCreate):
                print("RefCreate node exists")
            #     res = self.visit_ref_create(node)
            elif isinstance(node, RefRead):
                print("RefRead node exists")
            #     res = self.visit_ref_read(node)
            elif isinstance(node, RefWrite):
                print("RefWrite node exists")
            #     res = self.visit_ref_write(node)
            elif isinstance(node, Constructor):
                print("Construction node exists")
            #     res = self.visit_constructor(node)
            elif isinstance(node, Match):
                print("Match node exists")
            #     res = self.visit_match(node)
            else:
                pass
                # raise Exception(f"warning unhandled case: {type(node)}")
            
        dot.render(filename=self.outFileName)
        
        
resnet, params = testing.resnet.get_workload(num_layers=18)

dot = Digraph(format='pdf')
# dot.attr(rankdir='BT')
dot.attr('node', shape='box')

nodeDict = {}
visitor = MyVisitor(nodeDict)
visitor.visit(resnet["main"])

for node, nodeIdx in nodeDict.items():
    if isinstance(node, Function):
        funcName = ''
        for pair in resnet.functions_items():
            if pair[1] == node:
                funcName = pair[0]
                break
            
        dot.node(str(nodeIdx), f'Function: {funcName}')
        dot.edge(str(nodeDict[node.body]), str(nodeIdx))
        for param in node.params:
            dot.edge(str(nodeIdx), str(nodeDict[param]))
    elif isinstance(node, Call):
        args = [nodeDict[arg] for arg in node.args]
        dot.node(str(nodeIdx), f'Call(op={node.op.name})', fillcolor='green', style='filled')
        for arg in args:
            dot.edge(str(arg), str(nodeIdx))
    # elif isinstance(node, Let):
    elif isinstance(node, Var):
        dot.node(str(nodeIdx), f'{node.name_hint}:\nshape=[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]', fillcolor='yellow', style='filled')
    elif isinstance(node, GlobalVar):
        # maybe for tensors not function names
        dot.node(str(nodeIdx), f'{node.name_hint}:\nshape=[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]')
    elif isinstance(node, If):
        dot.node(str(nodeIdx), f'If')
        dot.edge(str(nodeDict[node.cond]), str(nodeIdx))
        dot.edge(str(nodeIdx), str(nodeDict[node.true_branch]), "TRUE")
        dot.edge(str(nodeIdx), str(nodeDict[node.false_branch]), "FALSE")
    elif isinstance(node, Tuple):
        dot.node(str(nodeIdx), f'Tuple')
        for field in node.fields:
            dot.edge(str(nodeDict[field], str(nodeIdx)))
    elif isinstance(node, TupleGetItem):
        dot.node(str(nodeIdx), f'TupleGetItem(idx={node.index})', fillcolor='red', style='filled')
        dot.edge(str(nodeDict[node.tuple_value]), str(nodeIdx))
    elif isinstance(node, Constant):
        dot.node(str(nodeIdx), f'Const: shape=[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]', fillcolor='blue', style='filled')
    elif isinstance(node, Op):
        print("Op node exists")
    #     dot.node(str(nodeIdx), f'{node.name}')
    elif isinstance(node, RefCreate):
        print("RefCreate node exists")
    #     res = self.visit_ref_create(node)
    elif isinstance(node, RefRead):
        print("RefRead node exists")
    #     res = self.visit_ref_read(node)
    elif isinstance(node, RefWrite):
        print("RefWrite node exists")
    #     res = self.visit_ref_write(node)
    elif isinstance(node, Constructor):
        print("Construction node exists")
    #     res = self.visit_constructor(node)
    elif isinstance(node, Match):
        print("Match node exists")
    #     res = self.visit_match(node)
    else:
        pass
        # raise Exception(f"warning unhandled case: {type(node)}")

dot.render(filename="resnet18")