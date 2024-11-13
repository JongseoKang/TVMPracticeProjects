import sys

sys.path.append("/home/2020147555/myenv/lib/python3.8/site-packages")
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

from visualiser import Visualiser


mod, params = testing.vgg.get_workload(batch_size=1, num_layers=19)

vgg = Visualiser(mod, "vgg19", "pdf")
vgg.run()