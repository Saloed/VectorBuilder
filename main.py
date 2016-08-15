from pycparser import parse_file

from TBCNN.Builder import construct_from_ast
from TBCNN.InitParams import init_prepared_params

ast = parse_file('test.c')
params = init_prepared_params()
network = construct_from_ast(ast, params)
