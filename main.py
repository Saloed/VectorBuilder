import theano
from pycparser import parse_file

from TBCNN.Builder import construct_from_ast
from TBCNN.InitParams import init_prepared_params
from TBCNN.Propagations import forward_propagation

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'
theano.config.mode = 'DebugMode'

ast = parse_file('test.cpp', use_cpp=True)

ast.show()

params = init_prepared_params()
network = construct_from_ast(ast, params)
result = forward_propagation(network)
print(result)
