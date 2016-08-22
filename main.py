import theano
from pycparser import parse_file
import theano.tensor as T
from theano import function
from TBCNN.Builder import construct_from_ast
from TBCNN.InitParams import init_prepared_params
from TBCNN.Propagations import forward_propagation

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'
theano.config.mode = 'DebugMode'
theano.config.floatX = 'float32'

ast = parse_file("test.cpp", use_cpp=True)

# ast.show()
params = init_prepared_params()
network = construct_from_ast(ast, params)
result = forward_propagation(network)
# print(result)

x = T.fmatrix('x')
y = T.fmatrix('y')
# dispersion = T.mean(T.pow(x - y, 2))
# var = T.var(x - y)
std = T.std(x - y)
error = function([x, y], std)
print("Test start")
results = []


# can't run others because of parser
def evaluate_file(filename):
    print("Building vector for " + filename)
    ast = parse_file(filename, use_cpp=True)
    network = construct_from_ast(ast, params)
    return forward_propagation(network)


for i in range(4):
    filename = "Dataset/test_" + str(i) + ".cpp"
    results.append(evaluate_file(filename))

# one_more_file = "Dataset/test_5.cpp"
# results.append(evaluate_file(one_more_file))
print()
for (i, r_i) in enumerate(results):
    for (j, r_j) in enumerate(results):
        if i != j:
            mse = error(r_i, r_j)
            print("Diff between test_" + str(i) + " and test_" + str(j) + " : ", mse)
    print()
