import _pickle as P
import gc
import math
import os
import sys
from itertools import cycle
from random import randint, shuffle

import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import sgd
from theano import function

from AST.GitAuthor import DataSet
from AST.GitAuthor import get_repo_methods_with_authors, get_single_author_data
from AuthorClassifier.Builder import NetLoss
from AuthorClassifier.Builder import construct_from_nodes, BuildMode
from AuthorClassifier.ClassifierParams import NUM_RETRY, NUM_EPOCH, l2_param, BATCH_SIZE, learn_rate, SAVE_PERIOD
from AuthorClassifier.InitParams import init_params
from Utils.Visualization import new_figure
from Utils.Visualization import save_to_file
from Utils.Visualization import update_figure
from Utils.Wrappers import safe_run, timing

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'


# theano.config.exception_verbosity = 'high'

def generate_author_file():
    data_set = get_repo_methods_with_authors('../Dataset/CombinedProjects/MPS/')
    # data_set = get_single_author_data('../Dataset/OneAuthorProjects/distributedlog/')
    with open('../Dataset/CombinedProjects/author_file_MPS', 'wb') as f:
        P.dump(data_set, f)


class Batch:
    def __init__(self, ast, author, index):
        self.ast = ast
        self.author = author
        self.index = index
        self.back = None

    def __str__(self):
        return '{} {} {}'.format(self.index, str(self.author), str(self.ast.root_node))

    def __repr__(self):
        return self.__str__()


def generate_batches(data: list, r_index) -> list:
    return [Batch(d, d.root_node.author, r_index[d.root_node.author]) for d in data]


log_file = open('log.txt', mode='w')


def fprint(print_str: list, file=log_file):
    for str in print_str:
        print(str)
        print(str, file=file)
    file.flush()


def build_vectors(authors):
    index = {}
    size = len(authors)
    # assuming that we have only two authors
    for uauthor in authors:
        for author in uauthor[1]:
            vec = [0] * size
            vec[uauthor[0]] = 1
            index[author] = (vec, uauthor[0])
    return size, index


@safe_run
def epoch_step(train_epoch, retry_num, train_fun, test_fun, nparams):
    @timing
    def process_set(set_fun):
        res = ([], [], [], [])
        for fun in set_fun:
            loss, loss_std, max_loss, err = fun()
            res[0].append(float(loss))
            res[1].append(float(loss_std))
            res[2].append(float(max_loss))
            res[3].append(float(err))
        loss = np.mean(res[0])
        loss_std = np.mean(res[1])
        loss_max = np.max(res[2])
        err = np.mean(res[3])
        return loss, loss_std, loss_max, err

    shuffle(train_fun)
    tr_loss, tr_std, tr_max, tr_err = process_set(train_fun)
    te_loss, te_std, te_max, te_err = process_set(test_fun)

    print_str = [
        'epoch {0} retry {1}'.format(train_epoch, retry_num),
        'train | mean {0:.4f} | std {1:.4f} | max {2:.4f} | percent {3:.2f}'.format(float(tr_loss), float(tr_std),
                                                                                    float(tr_max), float(tr_err)),
        'test  | mean {0:.4f} | std {1:.4f} | max {2:.4f} | percent {3:.2f}'.format(float(te_loss), float(te_std),
                                                                                    float(te_max), float(te_err)),
        '################'
    ]
    fprint(print_str, log_file)
    if math.isnan(tr_loss) or math.isnan(te_loss):
        raise Exception('Error is NAN. Start new retry')

    if train_epoch % SAVE_PERIOD == 0:
        with open('AuthorClassifier/NewParams/new_params_t' + str(retry_num) + "_ep" + str(train_epoch),
                  mode='wb') as new_params:
            P.dump(nparams, new_params)

    return te_loss, tr_loss


def reset_batches(batches):
    for batch in batches:
        batch.back = None
        # batch.valid = None
        # gc.collect()


def get_errors(batches, need_l2, l2):
    loss = [b.back.loss for b in batches]
    error = [b.back.error for b in batches]
    cost = T.mean(loss)
    if need_l2: cost = cost + l2
    err = T.mean(error)
    max_loss = T.max(loss)
    loss_std = T.std(T.as_tensor_variable(loss))

    return cost, loss_std, max_loss, err


def build_all(batches, nparams, r_index, authors_amount):
    for i, batch in enumerate(batches):
        author = r_index[batch.author]
        batch.back = construct_from_nodes(batch.ast, nparams, BuildMode.train, author, authors_amount)  # type: NetLoss


def init_set(train_set, test_set, nparams, r_index, authors_amount):
    train_batches = [train_set[i:i + BATCH_SIZE] for i in range(0, len(train_set), BATCH_SIZE)]


    @timing
    def build_test_batch(batch):
        test_loss, test_loss_std, test_max_loss, test_err = get_errors(batch, False, None)
        return function([], [test_loss, test_loss_std, test_max_loss, test_err])

    fprint(['test set'])
    build_all(test_set, nparams, r_index, authors_amount)

    test_batches = [test_set[i:i + BATCH_SIZE] for i in range(0, len(test_set), BATCH_SIZE)]
    test_fun = [build_test_batch(b) for b in test_batches]
    reset_batches(test_set)

    # train_batches = cycle(train_batches)

    return train_batches, test_fun


@timing
def build_train_fun(batch, nparams, r_index, authors_amount):
    build_all(batch, nparams, r_index, authors_amount)

    weights = list(nparams.w.values())
    bias = list(nparams.b.values())
    squared = [T.sqr(p).sum() for p in weights]
    l2 = l2_param * T.sum(squared)

    train_loss, train_loss_std, train_max_loss, train_err = get_errors(batch, True, l2)
    updates = sgd(train_loss, weights + bias, learn_rate)
    train_fun = function([], [train_loss, train_loss_std, train_max_loss, train_err], updates=updates)
    reset_batches(batch)
    return train_fun


# @safe_run
def train_step(retry_num, train_set, test_set, authors):
    reset_batches(train_set)
    reset_batches(test_set)
    authors_amount, r_index = build_vectors(authors)
    nparams = init_params(authors_amount, 'AuthorClassifier/emb_params')
    train_batches, test_fun = init_set(train_set, test_set, nparams, r_index, authors_amount)
    train_fun = []
    for train_batch in train_batches:
        fun = build_train_fun(train_batch, nparams, r_index, authors_amount)
        train_fun.append(fun)
        gc.collect()

    plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 1)  # len(authors) + 1)

    for train_epoch in range(NUM_EPOCH):
        error = epoch_step(train_epoch, retry_num, train_fun, test_fun, nparams)
        if error is None:
            break
        verr, terr = error
        update_figure(plot, plot_axes, train_epoch, verr, terr)

    save_to_file(plot, 'retry{}.png'.format(retry_num))


def prepare_batches(batches):
    new_batches = {}
    for k, v in batches.items():
        samples = [s for s in v if len(s.ast.all_nodes) > 150]
        shuffle(samples)
        new_batches[k] = samples
    return new_batches


def divide_data_set(data_set, train_units, test_units):
    train_set = []
    test_set = []

    for i in range(train_units):
        for k, itm in data_set.items():
            size = len(itm)
            if i > size - test_units:
                pos = randint(0, size - test_units)
                train_set.append(itm[pos])
            else:
                train_set.append(itm[i])

    for i in range(test_units):
        for k, itm in data_set.items():
            test_set.append(itm[len(itm) - 1 - i])

    return train_set, test_set


def spec_main():
    sys.setrecursionlimit(99999)
    np.set_printoptions(threshold=100000)
    gc.enable()
    with open('Dataset/CombinedProjects/top_authors_MPS', 'rb') as f:
        # with open('Dataset/author_file_kylin', 'rb') as f:
        dataset = P.load(f)
    dataset = dataset[:5]
    indexes = range(len(dataset))
    r_index = {aa: i for i, a in enumerate(dataset) for aa in a[1]}
    authors = [(i, dataset[i][1]) for i in indexes]
    gc.collect()
    batches = {i: generate_batches(dataset[i][0], r_index) for i in indexes}
    batches = prepare_batches(batches)
    train_set, test_set = divide_data_set(batches, 40, 20)
    dataset, batches = (None, None)
    gc.collect()
    for train_retry in range(NUM_RETRY):
        train_step(train_retry, train_set, test_set, authors)


def main():
    sys.setrecursionlimit(99999)
    np.set_printoptions(threshold=100000)
    gc.enable()
    dataset: list[DataSet] = [None, None]
    with open('Dataset/CombinedProjects/author_file_compass', 'rb') as f:
        dataset[0] = P.load(f)  # type: DataSet
    with open('Dataset/CombinedProjects/author_file_textmapper', 'rb') as f:
        dataset[1] = P.load(f)  # type: DataSet

    r_index = {a: i for i in [0, 1] for a in dataset[i].all_authors}
    authors = [(0, dataset[0].all_authors), (1, dataset[1].all_authors)]

    batches = {0: generate_batches(dataset[0].methods_with_authors, r_index),
               1: generate_batches(dataset[1].methods_with_authors, r_index)}

    batches = prepare_batches(batches)
    train_set, test_set = divide_data_set(batches, 2, 2)

    for train_retry in range(NUM_RETRY):
        train_step(train_retry, train_set, test_set, authors)


if __name__ == '__main__':
    # main()
    spec_main()
    # generate_author_file()
    # test()
