import _pickle as P
import gc
import sys
from collections import namedtuple
from copy import deepcopy
from itertools import groupby
from random import randint, shuffle
import math
import numpy as np
import theano

from AST.Tokenizer import print_ast
from AuthorClassifier.ClassifierParams import NUM_RETRY, NUM_EPOCH

from AST.GitAuthor import get_repo_methods_with_authors
from AuthorClassifier.Builder import construct_from_nodes, BuildMode
from AuthorClassifier.InitParams import init_params, reset_params
from Utils.Visualization import new_figure, update_figure, save_to_file
from Utils.Wrappers import safe_run

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'


# theano.config.exception_verbosity = 'high'

def generate_author_file():
    data_set = get_repo_methods_with_authors('../Dataset/TestRepo/')
    with open('../Dataset/author_file', 'wb') as f:
        P.dump(data_set, f)


class Batch:
    def __init__(self, ast, author, index):
        self.ast = ast
        self.author = author
        self.index = index
        self.valid = None
        self.back = None
        self.back_svm = None

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
    for uauthor in authors:
        for author in uauthor[1]:
            one_hot = np.ones(size, dtype='float32')
            one_hot *= -1.0
            # one_hot = np.zeros(size, dtype='float32')
            one_hot[uauthor[0]] = 1
            index[author] = one_hot
    return index


def process_set(batches, nparams, need_back, authors):
    err = 0
    rerr = 0
    size = len(batches)
    author_amount = len(authors)
    reverse_index = build_vectors(authors)
    for i, batch in enumerate(batches):
        author = reverse_index[batch.author]
        if need_back:
            if batch.back is None or batch.back_svm is None:
                fprint(['build {}'.format(i)])
                batch.back, batch.back_svm = construct_from_nodes(batch.ast, nparams, BuildMode.train, author_amount)
            if i < 0.7 * size:
                terr, e, res = batch.back(author)
            else:
                terr, e, res = batch.back_svm(author)
            fprint([batch.author, author, res, terr, e])
            rerr += e
            err += terr
        else:
            if batch.valid is None:
                fprint(['build {}'.format(i)])
                batch.valid = construct_from_nodes(batch.ast, nparams, BuildMode.validation, author_amount)
            terr, e, res = batch.valid(author)
            fprint([batch.author, author, res, terr, e])
            rerr += e
            err += terr
        # fprint([nparams.w['w_conv_root'].eval(), nparams.b['b_conv'].eval()])
        if math.isnan(terr):
            raise Exception('Error is NAN. Start new retry')
    return err / size, rerr / size


@safe_run
def epoch_step(nparams, train_epoch, retry_num, batches, test_set, authors):
    shuffle(batches)
    fprint(['train set'])
    result = process_set(batches, nparams, True, authors)
    if result is None:
        return
    tr_err, tr_rerr = result
    fprint(['test set'])
    result = process_set(test_set, nparams, False, authors)
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
        with open('NewParams/new_params_t' + str(retry_num) + "_ep" + str(train_epoch), mode='wb') as new_params:
            P.dump(nparams, new_params)

    return test_err, tr_err


def reset_batches(batches):
    for batch in batches:
        batch.back = None
        batch.back_svm = None
        batch.valid = None
    gc.collect()


@safe_run
def train_step(retry_num, batches, test_set, authors, nparams):
    init_params(authors, 'emb_params')
    reset_batches(batches)
    reset_batches(test_set)
    plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 4)  # len(authors) + 1)
    for train_epoch in range(NUM_EPOCH):
        error = epoch_step(nparams, train_epoch, retry_num, batches, test_set, authors)
        if error is None:
            return
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
        indexed[d.index].append(d)

    for k in indexed.keys():
        batches = indexed[k]
        if len(batches) < 700:
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
    with open('../Dataset/author_file', 'rb') as f:
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
    with open('../Dataset/author_file', 'rb') as f:
        dataset = P.load(f)

    sys.setrecursionlimit(99999)
    np.set_printoptions(threshold=100000)

    all_authors = dataset.all_authors
    authors, r_index = collapse_authors(all_authors)
    all_batches = generate_batches(dataset.methods_with_authors, r_index)
    batches, r_index, authors = group_batches(all_batches, r_index, authors)
    train_set, test_set = divide_data_set(batches, 200, 100)
    nparams = init_params(authors, 'emb_params')
    for train_retry in range(NUM_RETRY):
        train_step(train_retry, train_set, test_set, authors, nparams)


if __name__ == '__main__':
    main()
    # generate_author_file()
    # test()
