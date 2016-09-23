import theano
import _pickle as P
import theano.tensor as T
from theano import function

from AST.Tokenizer import build_ast
from TBCNN.Builder import construct_from_nodes
from TBCNN.InitParams import init_prepared_params, rand_params

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'

ast = build_ast("test.java")

params = rand_params()

network = construct_from_nodes(ast, params, need_back_prop=True)
