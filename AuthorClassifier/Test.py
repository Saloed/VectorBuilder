import sys
import _pickle as P

import gc

from theano import function

from AST.GitAuthor import DataSet
from AuthorClassifier.Builder import NetLoss
from AuthorClassifier.Builder import construct_from_nodes
from AuthorClassifier.Builder import BuildMode
import numpy as np

from AuthorClassifier.Train import generate_batches, prepare_batches, build_vectors, divide_data_set


def process_batches(batches, authors, nparams):
    r_index = build_vectors(authors)
    error = 0
    size = len(batches)
    for i, batch in enumerate(batches):
        author = r_index[batch.author]
        net = construct_from_nodes(batch.ast, nparams, BuildMode.validation, author)  # type: NetLoss
        fun = function([], net.error)
        err = fun()[0]
        print(err, error)
        error += err
    return error / size


def main():
    sys.setrecursionlimit(99999)
    np.set_printoptions(threshold=100000)
    gc.enable()

    with open('Dataset/author_file_AndEngine', 'rb') as f:
        dataset_0 = P.load(f)  # type: DataSet
    with open('Dataset/author_file_distributedlog', 'rb') as f:
        dataset_1 = P.load(f)  # type: DataSet
    with open('AuthorClassifier/best_result', 'rb') as f:
        params = P.load(f)

    r_index = {dataset_0.all_authors[0]: 0, dataset_1.all_authors[0]: 1}
    authors = [(0, [dataset_0.all_authors[0]]), (1, [dataset_1.all_authors[0]])]

    batches = {0: generate_batches(dataset_0.methods_with_authors, r_index),
               1: generate_batches(dataset_1.methods_with_authors, r_index)}

    batches = prepare_batches(batches)
    _, batches = divide_data_set(batches, 0, 900)

    res = process_batches(batches, authors, params)
    print('all data mean error {}'.format(res))


if __name__ == '__main__':
    main()
