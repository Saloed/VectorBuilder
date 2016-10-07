import _pickle as P
from collections import namedtuple

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


def generate_batches(data: list) -> list:
    return [Batch(d, d.root_node.author) for d in data]


log_file = open('log.txt', mode='w')


def fprint(print_str: list, file=log_file):
    for str in print_str:
        print(str)
        print(str, file=file)
    file.flush()


def process_set(batches, nparams, need_back, authors):
    err = 0
    size = len(batches)
    author_amount = len(list(authors.values()))
    for batch in batches:
        author = authors[batch.author]
        if need_back:
            if batch.back is None:
                batch.back = construct_from_nodes(batch.ast, nparams, need_back, author_amount)
            err += batch.back(author)
        else:
            if batch.valid is None:
                batch.valid = construct_from_nodes(batch.ast, nparams, need_back, author_amount)
            err += batch.valid(author)
    return err / size


# @safe_run
def epoch_step(nparams, train_epoch, retry_num, batches, test_set, authors: dict):
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


# @safe_run
def train_step(retry_num, batches, test_set, all_authors):
    nparams = init_params(all_authors)
    reset_batches(batches)
    reset_batches(test_set)
    plot_axes, plot = new_figure(retry_num, NUM_EPOCH)
    authors = {}
    for i, author in enumerate(all_authors):
        authors[author] = i
    for train_epoch in range(NUM_EPOCH):
        error = epoch_step(nparams, train_epoch, retry_num, batches, test_set, authors)
        if error is None:
            return
        update_figure(plot, plot_axes, train_epoch, error)

    save_to_file(plot, 'retry{}.png'.format(retry_num))


def main():
    with open('../Dataset/author_file', 'rb') as f:
        dataset = P.load(f)

    all_authors = dataset.all_authors
    batches = generate_batches(dataset.methods_with_authors)
    batches = batches[:1]
    test_set = batches[2:3]
    for train_retry in range(NUM_RETRY):
        train_step(train_retry, batches, test_set, all_authors)


if __name__ == '__main__':
    main()
    # generate_author_file()
