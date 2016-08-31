import os
import pickle
import theano
import theano.tensor as T
import time
from random import shuffle
from theano import function
import numpy as np
from copy import copy, deepcopy

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


def evaluate(positive: Network, negative: Network, training_token: Token,
             params: Parameters, alpha, is_validation):
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
    if is_validation:
        return error
    positive.back(pos_forward - pos_target, alpha)
    negative.back(neg_target - neg_forward, alpha)

    return error


def prepare_ast(full_ast):
    nodes_with_depth = []

    def compute_depth(node: Token, depth):
        if len(node.children) != 0:
            nodes_with_depth.append((node, depth))
            for child in node.children:
                compute_depth(child, depth + 1)

    compute_depth(full_ast[-1], 0)
    nodes_with_depth.sort(key=lambda tup: tup[1])
    nodes = [node[0] for node in reversed(nodes_with_depth)]

    class Indexer:
        def __init__(self):
            self.indexer = 0

        def children(self, node: Token, ast=None, parent=None) -> list:
            if ast is None:
                ast = []
            nd = copy(node)
            nd.parent = parent
            nd.pos = self.indexer
            ast.append(nd)
            self.indexer += 1
            for child in nd.children:
                self.children(child, ast, nd.pos)
            return ast

    return [Indexer().children(node) for node in nodes]


def build_update(params: Parameters, update: Updates):
    upd = []
    alpha = T.fscalar('alpha')

    for key, value in params.w.items():
        upd.append((value, value - alpha * update.grad_w[key]))
    for value in params.embeddings:
        key = value.name
        upd.append((value, value - alpha * update.grad_b[key]))
    return function([alpha], updates=upd)


def timing(f):
    def wrap(*args):
        # print('%s function start' % (f.__name__,))
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        elapse = (time2 - time1) * 1000.0
        seconds = elapse / 1000
        millis = elapse % 1000
        print('%s function elapse %i sec %i ms' % (f.__name__, seconds, millis))
        return ret

    return wrap


@timing
def process_ast(file_ast, params, update, train_error, alpha, is_validation):
    prepared = prepare_ast(file_ast)
    for ast in prepared:
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
            train_error += evaluate(positive, neg, training_token, params, alpha, is_validation)
    return train_error


@timing
def epoch(data_ast, params, update, update_func, alpha, is_validation):
    train_error = 0
    for file_ast in data_ast:
        train_error = process_ast(file_ast, params, update, train_error, alpha, is_validation)
        if not is_validation:
            update_func(alpha)
            update = Updates()
        print(train_error)
    return train_error


def main():
    dataset_dir = '../Dataset/'
    ast_file = open(dataset_dir + 'ast_file', mode='rb')
    data_ast = pickle.load(ast_file)
    for tr in range(10):
        data_set = deepcopy(data_ast)
        update = Updates()
        alpha = LEARN_RATE * (1 - MOMENTUM)
        previous_train_error = 0
        params = initialize()
        update_func = build_update(params, update)

        for ep in range(13):
            shuffle(data_set)
            train_set = deepcopy(data_set[:len(data_ast) - 100])
            validation_set = deepcopy(data_set[len(data_ast) - 100:])
            train_error = epoch(train_set, params, update, update_func, alpha, False)
            validation_error = epoch(validation_set, params, update, update_func, alpha, True)
            print('################')
            print('end of epoch')
            print('delta', train_error - previous_train_error)
            print('validation', validation_error)
            print('new epoch')
            if train_error > previous_train_error * 2:
                alpha *= 0.95
            else:
                alpha *= 0.999
            previous_train_error = train_error

            new_params = open('new_params_t' + tr + '_ep' + ep, mode='wb')
            pickle.dump(params, new_params)


if __name__ == '__main__':
    main()
    # build_asts('../Dataset/')
