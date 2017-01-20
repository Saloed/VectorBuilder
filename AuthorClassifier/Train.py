import _pickle as P
import gc
import math
import sys
from itertools import cycle
from random import randint, shuffle

import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import sgd
from theano import function

from AST.GitAuthor import get_repo_methods_with_authors
from AuthorClassifier.Builder import NetLoss
from AuthorClassifier.Builder import construct_from_nodes, BuildMode
from AuthorClassifier.ClassifierParams import NUM_RETRY, NUM_EPOCH, l2_param, BATCH_SIZE, learn_rate
from AuthorClassifier.InitParams import init_params
from Utils.Visualization import new_figure
from Utils.Visualization import save_to_file
from Utils.Visualization import update_figure
from Utils.Wrappers import safe_run, timing

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'


# theano.config.exception_verbosity = 'high'

def generate_author_file():
    data_set = get_repo_methods_with_authors('Dataset/TestRepo/')
    with open('Dataset/author_file', 'wb') as f:
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
            index[author] = uauthor[0]
    return index


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

    if train_epoch % 100 == 0:
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


def build_all(batches, nparams, r_index):
    for i, batch in enumerate(batches):
        author = r_index[batch.author]
        batch.back = construct_from_nodes(batch.ast, nparams, BuildMode.train, author)  # type: NetLoss


def init_set(train_set, test_set, nparams, r_index):
    train_batches = [train_set[i:i + BATCH_SIZE] for i in range(0, len(train_set), BATCH_SIZE)]

    @timing
    def build_test_batch(batch):
        test_loss, test_loss_std, test_max_loss, test_err = get_errors(batch, False, None)
        return function([], [test_loss, test_loss_std, test_max_loss, test_err])

    fprint(['test set'])
    build_all(test_set, nparams, r_index)

    test_batches = [test_set[i:i + BATCH_SIZE] for i in range(0, len(test_set), BATCH_SIZE)]
    test_fun = [build_test_batch(b) for b in test_batches]
    reset_batches(test_set)

    train_batches = cycle(train_batches)

    return train_batches, test_fun


@timing
def build_train_fun(batch, nparams, r_index):
    build_all(batch, nparams, r_index)

    weights = list(nparams.w.values())
    bias = list(nparams.b.values())
    squared = [T.sqr(p).sum() for p in weights]
    l2 = l2_param * T.sum(squared)

    train_loss, train_loss_std, train_max_loss, train_err = get_errors(batch, True, l2)
    updates = sgd(train_loss, weights + bias, learn_rate)
    train_fun = [function([], [train_loss, train_loss_std, train_max_loss, train_err], updates=updates)]
    reset_batches(batch)
    return train_fun


# @safe_run
def train_step(retry_num, train_set, test_set, authors, nparams):
    nparams = init_params(authors, 'AuthorClassifier/emb_params')
    reset_batches(train_set)
    reset_batches(test_set)
    r_index = build_vectors(authors)
    train_batches, test_fun = init_set(train_set, test_set, nparams, r_index)

    plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 1)  # len(authors) + 1)

    for train_epoch in range(NUM_EPOCH):

        if train_epoch % 10 == 0:
            train_fun = None
            gc.collect()
            train_fun = build_train_fun(next(train_batches), nparams, r_index)

        error = epoch_step(train_epoch, retry_num, train_fun, test_fun, nparams)
        if error is None:
            break
        verr, terr = error
        update_figure(plot, plot_axes, train_epoch, verr, terr)

    save_to_file(plot, 'retry{}.png'.format(retry_num))


def collapse_authors(authors: list):
    unique_authors = []
    synonym = ['Li, Yang', 'liyang@apache.org']

    def search_for_new(author):
        for uauthor in unique_authors:
            for a in uauthor[1]:
                if author.name in synonym and a.name in synonym:
                    uauthor[1].append(author)
                    return False

                if a.name == author.name or a.email == author.email:
                    uauthor[1].append(author)
                    return False
        return True

    for a in authors:
        if search_for_new(a):
            unique_authors.append((len(unique_authors), [a]))
    revers_index = {}
    for ua in unique_authors:
        index = ua[0]
        for a in ua[1]:
            revers_index[a] = index
    return unique_authors, revers_index


def group_batches(data, r_index, authors):
    indexed = {}
    for d in data:
        if d.index not in indexed:
            indexed[d.index] = []
        if len(d.ast.all_nodes) > 100:
            indexed[d.index].append(d)

    for k in indexed.keys():
        batches = indexed[k]
        if len(batches) < 600:
            for b in batches:
                if b.author in r_index:
                    r_index[b.author] = None
            indexed[k] = None

    index = 0
    new_index = {}
    for k, itm in indexed.items():
        if itm is not None:
            for i in itm:
                i.index = index
                r_index[i.author] = index
            new_index[index] = itm
            index += 1
    r_index = {k: v for k, v in r_index.items() if v is not None}
    authors = {}
    for k, v in r_index.items():
        if v not in authors:
            authors[v] = []
        authors[v].append(k)
    authors = [(k, v) for k, v in authors.items()]
    return new_index, r_index, authors


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


def test():
    with open('Dataset/author_file', 'rb') as f:
        dataset = P.load(f)

    sys.setrecursionlimit(99999)

    all_authors = dataset.all_authors
    authors, r_index = collapse_authors(all_authors)
    all_batches = generate_batches(dataset.methods_with_authors, r_index)
    batches, r_index, authors = group_batches(all_batches, r_index, authors)
    train_set, test_set = divide_data_set(batches, 100, 200)
    batches = train_set + test_set

    with open('test_params', 'rb') as f:
        nparams = P.load(f)

    err = 0
    author_err = 0
    size = len(batches)
    author_amount = len(authors)
    reverse_index = build_vectors(authors)

    for i, batch in enumerate(batches):
        author = reverse_index[batch.author]
        fprint(['build {}'.format(i)])
        batch.valid = construct_from_nodes(batch.ast, nparams, BuildMode.validation, author_amount)
        terr, e, res = batch.valid(author)
        author_err += e
        fprint([batch.author, author, res, terr, e])
        err += terr
        if math.isnan(terr):
            raise Exception('Error is NAN. Start new retry')
    print('author recognition error {} {}\nmean cross entropy {} {}'.format(author_err, author_err / size, err,
                                                                            err / size))


def main():
    with open('Dataset/author_file', 'rb') as f:
        dataset = P.load(f)

    sys.setrecursionlimit(99999)
    np.set_printoptions(threshold=100000)

    all_authors = dataset.all_authors
    authors, r_index = collapse_authors(all_authors)
    all_batches = generate_batches(dataset.methods_with_authors, r_index)
    batches, r_index, authors = group_batches(all_batches, r_index, authors)
    train_set, test_set = divide_data_set(batches, 500, 50)
    nparams = init_params(authors, 'AuthorClassifier/emb_params')
    for train_retry in range(NUM_RETRY):
        train_step(train_retry, train_set, test_set, authors, nparams)


if __name__ == '__main__':
    main()
    # generate_author_file()
    # test()
