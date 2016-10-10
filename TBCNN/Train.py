import _pickle as P
from collections import namedtuple
import numpy as np
import gc
from random import shuffle

import theano

from AST.GitAuthor import get_repo_methods_with_authors
from TBCNN.Builder import construct_from_nodes
from TBCNN.InitParams import init_params
from TBCNN.NetworkParams import NUM_RETRY, NUM_EPOCH
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
            one_hot = np.zeros(size, dtype='int32')
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
                batch.back = construct_from_nodes(batch.ast, nparams, need_back, author_amount)
            err += batch.back(author)
        else:
            if batch.valid is None:
                print('build {}'.format(i))
                batch.valid = construct_from_nodes(batch.ast, nparams, need_back, author_amount)
            err += batch.valid(author)
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
    plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 10.0)
    for train_epoch in range(NUM_EPOCH):
        error = epoch_step(nparams, train_epoch, retry_num, batches, test_set, authors)
        if error is None:
            return
        update_figure(plot, plot_axes, train_epoch, error)

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
    return unique_authors


def main():
    with open('../Dataset/author_file', 'rb') as f:
        dataset = P.load(f)

    all_authors = dataset.all_authors
    authors = collapse_authors(all_authors)
    all_batches = generate_batches(dataset.methods_with_authors)
    shuffle(all_batches)
    batches = all_batches[:500]
    test_set = all_batches[500:600]
    for train_retry in range(NUM_RETRY):
        train_step(train_retry, batches, test_set, authors)


if __name__ == '__main__':
    main()
    # generate_author_file()
