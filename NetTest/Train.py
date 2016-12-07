import sys
import pickle as P
from copy import deepcopy
from random import shuffle

import gc
import numpy as np
import math

import theano

from AST.Tokenizer import Nodes
from AuthorClassifier.Builder import construct_from_nodes, BuildMode
from AuthorClassifier.InitParams import init_params
from Utils.Visualization import save_to_file, update_figure, new_figure
from Utils.Wrappers import timing, safe_run

log_file = open('log.txt', mode='w')
NUM_EPOCH = 1000
NUM_RETRY = 10
theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'


def fprint(print_str: list, file=log_file):
    for str in print_str:
        print(str)
        print(str, file=file)
    file.flush()


class Batch:
    def __init__(self, ast, has_cycle):
        self.ast = ast
        self.has_cycle = has_cycle
        self.valid = None
        self.back = None

    def __str__(self):
        return '{}'.format(str(self.has_cycle))

    def __repr__(self):
        return self.__str__()


@safe_run
def train_step(retry_num, batches, test_set, nparams):
    nparams = init_params([], 'AuthorClassifier/emb_params')
    reset_batches(batches)
    reset_batches(test_set)
    plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 1)  # len(authors) + 1)
    for train_epoch in range(NUM_EPOCH):
        error = epoch_step(nparams, train_epoch, retry_num, batches, test_set)
        if error is None:
            break
        verr, terr = error
        update_figure(plot, plot_axes, train_epoch, verr, terr)

    save_to_file(plot, 'retry{}.png'.format(retry_num))


@timing
def process_set(batches, nparams, need_back):
    err = 0
    rerr = 0
    size = len(batches)

    for i, batch in enumerate(batches):

        if need_back:
            if batch.back is None:
                fprint(['build {}'.format(i)])
                batch.back = construct_from_nodes(batch.ast, nparams, BuildMode.train, 1)
            terr, e, res = batch.back(batch.has_cycle)
            # fprint([batch.has_cycle, res, terr, e])
            rerr += e
            err += terr
        else:
            if batch.valid is None:
                fprint(['build {}'.format(i)])
                batch.valid = construct_from_nodes(batch.ast, nparams, BuildMode.validation, 1)
            terr, e, res = batch.valid(batch.has_cycle)
            # fprint([batch.has_cycle, res, terr, e])
            rerr += e
            err += terr
        # fprint([nparams.w['w_conv_root'].eval(), nparams.b['b_conv'].eval()])
        if math.isnan(terr):
            raise Exception('Error is NAN. Start new retry')
    return err / size, rerr / size


@safe_run
def epoch_step(nparams, train_epoch, retry_num, batches, test_set):
    shuffle(batches)
    fprint(['train set'])
    result = process_set(batches, nparams, True)
    if result is None:
        return
    tr_err, tr_rerr = result
    fprint(['test set'])
    result = process_set(test_set, nparams, False)
    if result is None:
        return
    test_err, test_rerr = result

    print_str = [
        'end of epoch {0} retry {1}'.format(train_epoch, retry_num),
        'train\t|\t{}|\t{}'.format(tr_err, tr_rerr),
        'test\t|\t{}|\t{}'.format(test_err, test_rerr),
        '################'
    ]
    fprint(print_str, log_file)

    if train_epoch % 100 == 0:
        with open('NetTest/NewParams/new_params_t' + str(retry_num) + "_ep" + str(train_epoch),
                  mode='wb') as new_params:
            P.dump(nparams, new_params)

    return test_err, tr_err


def check_cycle(ast: Nodes):
    for node in ast.all_nodes:
        if node.token_type in ['FOR_KEYWORD', 'FOREACH_STATEMENT', 'FOR_STATEMENT', 'WHILE_KEYWORD', 'WHILE_STATEMENT',
                               'DO_WHILE_STATEMENT']:
            return 1
    return 0


def reset_batches(batches):
    for batch in batches:
        batch.back = None
        batch.valid = None
    gc.collect()


def make_pairs(methods):
    marked = [(m, check_cycle(m)) for m in methods]
    marked.sort(key=lambda m: m[1])
    pairs = []
    size = len(marked) - 1
    i = 0
    while marked[i][1] == 0 and marked[size - i][1] == 1:
        pairs.append((marked[i], marked[size - i]))
        i += 1
    return pairs


def divide_data_set(methods_with_authors, train_size, test_size):
    methods = deepcopy(methods_with_authors)
    pairs = make_pairs(methods)
    size = len(pairs)
    tr_set = []
    te_set = []
    for i in range(train_size):
        tr_set.append(Batch(pairs[i][0][0], pairs[i][0][1]))
        tr_set.append(Batch(pairs[i][1][0], pairs[i][1][1]))
    for i in range(size - 1, size - test_size, -1):
        te_set.append(Batch(pairs[i][0][0], pairs[i][0][1]))
        te_set.append(Batch(pairs[i][1][0], pairs[i][1][1]))
    return tr_set, te_set


def main():
    with open('Dataset/author_file', 'rb') as f:
        dataset = P.load(f)

    sys.setrecursionlimit(99999)
    np.set_printoptions(threshold=100000)
    train_set, test_set = divide_data_set(dataset.methods_with_authors, 50, 25)
    nparams = init_params([], 'AuthorClassifier/emb_params')
    for train_retry in range(NUM_RETRY):
        train_step(train_retry, train_set, test_set, nparams)


if __name__ == '__main__':
    main()
