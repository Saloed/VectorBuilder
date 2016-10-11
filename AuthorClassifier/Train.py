import _pickle as P
import gc
import sys
from copy import deepcopy
from random import shuffle

import numpy as np
import theano
from AuthorClassifier.ClassifierParams import NUM_RETRY, NUM_EPOCH

from AST.GitAuthor import get_repo_methods_with_authors
from AuthorClassifier.Builder import construct_from_nodes, BuildMode
from AuthorClassifier.InitParams import init_params
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
    def __init__(self, ast, author):
        self.ast = ast
        self.author = author
        self.valid = None
        self.back = None

    def __str__(self):
        return '{} {}'.format(str(self.author), str(self.ast.root_node))

    def __repr__(self):
        return self.__str__()


def generate_batches(data: list) -> list:
    return [Batch(d, d.root_node.author) for d in data]


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
            one_hot = np.ones(size, dtype='int32')
            one_hot *= -1
            one_hot[uauthor[0]] = 1
            index[author] = one_hot
    return index


def process_set(batches, nparams, need_back, authors):
    err = 0
    size = len(batches)
    author_amount = len(authors)
    reverse_index = build_vectors(authors)
    for i, batch in enumerate(batches):
        author = reverse_index[batch.author]
        if need_back:
            if batch.back is None:
                print('build {}'.format(i))
                batch.back = construct_from_nodes(batch.ast, nparams, BuildMode.train, author_amount)
            terr, res = batch.back(author)
            print(batch.author, author, res, terr)
            err += terr
        else:
            if batch.valid is None:
                print('build {}'.format(i))
                batch.valid = construct_from_nodes(batch.ast, nparams, BuildMode.validation, author_amount)
            terr, res = batch.valid(author)
            print(batch.author, author, res, terr)
            err += terr
    return err / size


@safe_run
def epoch_step(nparams, train_epoch, retry_num, batches, test_set, authors):
    shuffle(batches)
    fprint(['train set'])
    result = process_set(batches, nparams, True, authors)
    if result is None:
        return
    tr_err = result
    fprint(['test set'])
    result = process_set(test_set, nparams, False, authors)
    if result is None:
        return
    test_err = result

    print_str = [
        'end of epoch {0} retry {1}'.format(train_epoch, retry_num),
        'train\t|\t{}'.format(tr_err),
        'test\t|\t{}'.format(test_err),
        '################'
    ]
    fprint(print_str, log_file)

    if train_epoch % 100 == 0:
        with open('NewParams/new_params_t' + str(retry_num) + "_ep" + str(train_epoch), mode='wb') as new_params:
            P.dump(nparams, new_params)

    return test_err


def reset_batches(batches):
    for batch in batches:
        batch.back = None
        batch.valid = None
    gc.collect()


@safe_run
def train_step(retry_num, batches, test_set, authors):
    nparams = init_params(authors)
    reset_batches(batches)
    reset_batches(test_set)
    # plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 10.0)
    for train_epoch in range(NUM_EPOCH):
        error = epoch_step(nparams, train_epoch, retry_num, batches, test_set, authors)
        if error is None:
            return
            # update_figure(plot, plot_axes, train_epoch, error)

            # save_to_file(plot, 'retry{}.png'.format(retry_num))


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
    return unique_authors


def find_author(data, start_index, author, size) -> Batch:
    for i in range(start_index, size):
        if data[i] is not None:
            if data[i].author == author:
                result = deepcopy(data[i])
                data[i] = None
                return result
    return None


def divide_data_set(data_set):
    data = deepcopy(data_set)
    size = len(data)
    result_set = []

    for i in range(size):
        if data[i] is not None:
            author = data[i].author
            test_pair = find_author(data, i + 1, author, size)
            if test_pair is not None:
                train_pair = deepcopy(data[i])
                data[i] = None
                result_set.append((train_pair, test_pair))
    shuffle(result_set)
    train_set = [pair[0] for pair in result_set]
    test_set = [pair[1] for pair in result_set]

    return train_set[:100], test_set[:20]


def main():
    with open('../Dataset/author_file', 'rb') as f:
        dataset = P.load(f)

    sys.setrecursionlimit(99999)

    all_authors = dataset.all_authors
    authors = collapse_authors(all_authors)
    all_batches = generate_batches(dataset.methods_with_authors)

    train_set, test_set = divide_data_set(all_batches)

    for train_retry in range(NUM_RETRY):
        train_step(train_retry, train_set, test_set, authors)


if __name__ == '__main__':
    main()
    # generate_author_file()
