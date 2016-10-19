import _pickle as P

from AST.Tokenizer import parser_init, build_ast, divide_by_methods, get_all_available_tokens, visualize
from AuthorClassifier.Builder import BuildMode
from AuthorClassifier.ConstructionTest.Builder import construct_from_nodes
from AuthorClassifier.InitParams import init_params


def create_ast():
    parser = parser_init()
    ast = build_ast('/home/sobol/PycharmProjects/VectorBuilder/AuthorClassifier/ConstructionTest/test.java', parser)
    methods = divide_by_methods(ast)
    print(methods[0])
    with open('javafile', 'wb') as f:
        P.dump(methods[0], f)


def load_ast():
    with open('javafile', 'rb') as f:
        return P.load(f)


def main():
    ast = load_ast()
    nparams = init_params([1, 2], '../emb_params')
    print(ast)
    construct_from_nodes(ast, nparams, BuildMode.train, 2)


if __name__ == '__main__':
    # create_ast()
    main()
    # visualize(load_ast().root_node,'ast.jpg')
