import os
import pickle
import theano
import theano.tensor as T
from theano import function
import numpy as np
from copy import copy

from AST.Token import Token
from AST.Tokenizer import build_ast
from AST.TokenMap import token_map
from Embeddings import Parameters
from Embeddings.InitEmbeddings import initialize
from Embeddings.Construct import construct

# theano.config.exception_verbosity = 'high'
# theano.config.mode = 'DebugMode'
from Embeddings.Parameters import *
from TBCNN.NetworkParams import Network

theano.config.optimizer = 'fast_compile'
theano.config.floatX = 'float32'

x = T.fvector('x')
mse = function(inputs=[x], outputs=T.mul(T.sum(T.mul(x, x)), 0.5))


def build_asts(dataset_dir):
    base_dir = dataset_dir
    dataset_dir += 'java_files/'
    files = os.listdir(dataset_dir)
    data_ast = []
    for file in files:
        try:
            ast = build_ast(dataset_dir + file)
            data_ast.append(ast)
        except Exception:
            continue
    ast_file = open(base_dir + 'ast_file', mode='wb')
    pickle.dump(data_ast, ast_file)


def evaluate(positive: Network, negative: Network, training_token: Token, params: Parameters):
    pos_forward = positive.forward()
    neg_forward = negative.forward()

    pos_target = params.embeddings[training_token.token_index].eval()
    neg_target = pos_target
    pos_diff = pos_forward - pos_target
    neg_diff = neg_forward - neg_target

    pos_mse = mse(pos_diff)
    neg_mse = mse(neg_diff)

    error = MARGIN + pos_mse - neg_mse
    if error < 0:
        return 0

    positive.back(pos_forward - pos_target)
    negative.back(neg_target - neg_forward)

    return error


def prepare_ast(ast):
    nodes_with_depth = []

    def compute_depth(node: Token, depth):
        if len(node.children) != 0:
            nodes_with_depth.append((node, depth))
            for child in node.children:
                compute_depth(child, depth + 1)

    compute_depth(ast[-1], 0)
    nodes_with_depth.sort(key=lambda tup: tup[1])
    nodes = [node[0] for node in reversed(nodes_with_depth)]

    def children(node, ast) -> list:
        for child in node.children:
            children(child, ast)
        ast.append(node)
        return ast

    prepared = [children(node, []) for node in nodes]
    return prepared


def build_update(params: Parameters, update: Updates):
    upd = []
    alpha = T.fscalar('alpha')

    for key, value in params.w.items():
        upd.append((value, value - alpha * update.grad_w[key]))
    for value in params.embeddings:
        key = value.name
        upd.append((value, value - alpha * update.grad_b[key]))
    return function([alpha], updates=upd)


def main():
    params = initialize()
    dataset_dir = '../Dataset/'
    ast_file = open(dataset_dir + 'ast_file', mode='rb')
    data_ast = pickle.load(ast_file)
    update = Updates()
    update_func = build_update(params, update)
    alpha = LEARN_RATE * (1 - MOMENTUM)
    for i in range(10):
        train_error = 0
        for file_ast in data_ast:
            for ast in prepare_ast(file_ast):
                training_token = ast[-1]
                positive = construct(ast, params, update)

                def rand_token():
                    return list(token_map.keys())[np.random.randint(0, len(token_map))]

                def create_negative(token_index):
                    current = ast[token_index]
                    new_token = rand_token()
                    while current.token_type == new_token:
                        new_token = rand_token()
                    sample = copy(ast)
                    new_token = Token(new_token, token_map[new_token], current.parent, current.pos)
                    new_token.children = current.children
                    new_token.children_num = current.children_num
                    sample[token_index] = new_token
                    return construct(sample, params, update)

                negative = [create_negative(i) for i in np.random.random_integers(0, len(ast) - 1, size=10)]

                for neg in negative:
                    train_error += evaluate(positive, neg, training_token, params)
                print(train_error)
        update_func(alpha)
        alpha *= 0.999

    new_params = open('new_params', mode='wb')
    pickle.dump(params, new_params)


if __name__ == '__main__':
    main()
    # build_asts('../Dataset/')
