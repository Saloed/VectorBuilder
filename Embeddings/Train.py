import os
import pickle
import theano
import theano.tensor as T
import time
from random import shuffle
from theano import function
import numpy as np
from copy import copy, deepcopy
from multiprocessing import Pool
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

    target = params.embeddings[training_token.token_index].eval()
    pos_diff = pos_forward - target
    neg_diff = neg_forward - target

    pos_mse = mse(pos_diff)
    neg_mse = mse(neg_diff)

    error = MARGIN + pos_mse - neg_mse

    if error < 0:
        return 0
    if is_validation:
        return error

    positive.back(target, neg_forward, alpha)
    negative.back(target, pos_forward, alpha)

    return error


# 0 because of preparing
training_token_index = 0


def timing(f):
    def wrap(*args):
        print('%s function start' % (f.__name__,))
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        elapse = (time2 - time1) * 1000.0
        seconds = elapse / 1000
        millis = elapse % 1000
        print('%s function elapse %i sec %i ms' % (f.__name__, seconds, millis))
        return ret

    return wrap


def prepare_ast(full_ast):
    nodes_with_depth = []

    def compute_depth(node: Token, depth):
        if len(node.children) != 0:
            nodes_with_depth.append((node, depth))
            for child in node.children:
                compute_depth(child, depth + 1)

    compute_depth(full_ast[-1], 0)
    nodes_with_depth.sort(key=lambda tup: tup[1])
    nodes = [deepcopy(node[0]) for node in reversed(nodes_with_depth)]

    class Indexer:
        def __init__(self):
            self.indexer = 0

        def children(self, node: Token, ast=None, parent=None) -> list:
            if ast is None:
                ast = []
                assert training_token_index == self.indexer
            node.parent = parent
            node.pos = self.indexer
            ast.append(node)
            self.indexer += 1
            for child in node.children:
                self.children(child, ast, node.pos)
            return ast

    return [Indexer().children(node) for node in nodes]


def build_update(params: Parameters, updates: list):
    upd = []
    alpha = T.fscalar('alpha')
    update = dict()
    for up in updates:
        for key, value in up.grad_w.items():
            if key in update:
                update[key] = update[key] + value
            else:
                update[key] = value
        for key, value in up.grad_b.items():
            if key in update:
                update[key] = update[key] + value
            else:
                update[key] = value
    for key, value in params.w.items():
        upd.append((value, value - alpha * update[key]))
    for value in params.embeddings:
        key = value.name
        upd.append((value, value - alpha * update[key]))
    return function([alpha], updates=upd)


class PreparedEvaluationSet:
    def __init__(self, positive: Network, negative: list, training_token, ast_len):
        self.positive = positive
        self.negative = negative
        self.training_token = training_token
        self.ast_len = ast_len


# @timing
def prepare_net(data, constructed_networks: list, params):
    prepared = prepare_ast(data)
    for ast in prepared:
        ast_len = len(ast)
        if ast_len < 3:
            continue
        training_token = ast[training_token_index]

        def rand_token():
            return list(token_map.keys())[np.random.randint(0, len(token_map))]

        def create_negative(token_index):
            sample = deepcopy(ast)
            current = sample[token_index]
            new_token = rand_token()
            while current.token_type == new_token:
                new_token = rand_token()
            current.token_type = new_token
            current.token_index = token_map[new_token]
            return sample

        # rand from 1 because root_token_index is 0
        samples = [create_negative(i) for i in np.random.random_integers(1, len(ast) - 1, size=2)]
        positive = construct(ast, params, training_token_index)
        negative = [construct(sample, params, training_token_index, True) for sample in samples]
        constructed_networks.append(PreparedEvaluationSet(positive, negative, training_token, ast_len))
    return len(prepared)


# num_process = os.cpu_count()
# pool = Pool(processes=num_process)


def prepare_networks(data_set, params) -> list:
    constructed_networks = []
    num_data = len(data_set)
    # num_data //= num_process
    # for batch in range(num_data):
    #     results = [
    #         pool.apply_async(prepare_net,
    #                          [
    #                              data_set[batch * i], constructed_networks, params
    #                          ])
    #         for i in range(num_process)
    #         ]
    #     for result in results:
    #         print('constructed ', result.get(), ' networks')
    num_rest = num_data
    for data in data_set:
        prepare_net(data, constructed_networks, params)
        num_rest -= 1
        print('constructed. rest ', num_rest)
    return constructed_networks


# @timing
def process_network(net: PreparedEvaluationSet, params, alpha, is_validation):
    train_error = 0
    for neg in net.negative:
        train_error += evaluate(net.positive, neg, net.training_token, params, alpha, is_validation)
    return train_error / net.ast_len, train_error


# @timing
def epoch(prepared_networks: list, params, alpha, is_validation):
    num_data = len(prepared_networks)
    train_error_per_token = 0
    train_error = 0

    for net in prepared_networks:
        error_per_token, temp_err = process_network(net, params, alpha, is_validation)
        train_error += temp_err
        train_error_per_token += error_per_token
        str = ['\t|\t{}\t|\t{}\t|\t{}'.format(error_per_token, temp_err, train_error)]
        fprint(str)

    return train_error / num_data, train_error_per_token / num_data


log_file = open('log.txt', mode='w')


def fprint(print_str: list, file=log_file):
    for str in print_str:
        print(str)
        print(str, file=file)
    file.flush()


def process_batch(batch, params, alpha, is_validation):
    try:
        nets = prepare_networks(batch, params)
    except Exception as exc:
        print('exception in net preparing')
        print(exc.__traceback__)
        return 0, 0

    try:
        error_per_ast, error_per_token = epoch(nets, params, alpha, is_validation)
    except Exception as exc:
        print('exception in epoch')
        print(exc.__traceback__)
        return 0, 0
    return error_per_ast, error_per_token


batch_size = 10


def main():
    dataset_dir = '../Dataset/'
    ast_file = open(dataset_dir + 'ast_file', mode='rb')
    data_ast = pickle.load(ast_file)
    for train_retry in range(20):
        data_set = deepcopy(data_ast)
        alpha = LEARN_RATE * (1 - MOMENTUM)
        prev_t_ast = 0
        prev_t_token = 0
        prev_v_ast = 0
        prev_v_token = 0

        params = initialize()
        try:
            for train_epoch in range(20):
                t_error_per_token = 0
                t_error_per_ast = 0
                v_error_per_token = 0
                v_error_per_ast = 0
                shuffle(data_set)
                train_set = deepcopy(data_set[:len(data_ast) - 100])
                validation_set = deepcopy(data_set[len(data_ast) - 100:])

                num_train = len(train_set) // batch_size
                for bat in range(num_train - 1):
                    temp_ast_e, temp_token_e = process_batch(
                        train_set[bat * batch_size:(bat + 1) * batch_size],
                        params, alpha, False)
                    t_error_per_ast += temp_ast_e
                    t_error_per_token += temp_token_e

                num_valid = len(validation_set) // batch_size
                for bat in range(num_valid - 1):
                    temp_ast_e, temp_token_e = process_batch(
                        validation_set[bat * batch_size:(bat + 1) * batch_size],
                        params, alpha, True)
                    v_error_per_ast += temp_ast_e
                    v_error_per_token += temp_token_e

                dtpt = prev_t_token - t_error_per_token
                dtpa = prev_t_ast - t_error_per_ast
                dvpt = prev_v_token - v_error_per_token
                dvpa = prev_v_ast - v_error_per_ast
                print_str = [
                    '################',
                    'end of epoch {0} retry {1}'.format(train_epoch, train_retry),
                    'train\t|\t{}\t|\t{}'.format(t_error_per_token, t_error_per_ast),
                    'delta\t|\t{}\t|\t{}'.format(dtpt, dtpa),
                    'validation\t|\t{}\t|\t{}'.format(v_error_per_token, v_error_per_ast),
                    'delta\t|\t{}\t|\t{}'.format(dvpt, dvpa),
                    '################',
                    'new epoch'
                ]
                fprint(print_str, log_file)
                alpha *= 0.999
                prev_t_token = t_error_per_token
                prev_t_ast = t_error_per_ast
                prev_v_token = v_error_per_token
                prev_v_ast = v_error_per_ast

                new_params = open('new_params_t' + str(train_retry) + "_ep" + str(train_epoch), mode='wb')
                pickle.dump(params, new_params)
        except Exception as exc:
            print('exception in epoch loop')
            print(exc.__traceback__)
            continue


if __name__ == '__main__':
    main()
    # build_asts('../Dataset/')
