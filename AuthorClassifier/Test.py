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
    author_amount, r_index = build_vectors(authors)
    error = 0
    size = len(batches)
    for i, batch in enumerate(batches):
        author = r_index[batch.author]
        net = construct_from_nodes(batch.ast, nparams, BuildMode.validation, author)  # type: NetLoss
        fun = function([], [net.net_forward, net.error])
        frwd, err = fun()
        frwd = float(frwd[0])
        err = err[0]
        print('err {0} frwd {1:.4} tar {2} total {3}'.format(err, frwd, author, error))
        error += err
    return error / size


def single_test():
    sys.setrecursionlimit(99999)
    np.set_printoptions(threshold=100000)
    gc.enable()
    with open('AuthorClassifier/best_result', 'rb') as f:
        params = P.load(f)
    with open('Dataset/CombinedProjects/author_file_MPS', 'rb') as f:
        dataset = P.load(f)  # type: DataSet
    # author_kimchy = dataset.all_authors[1]
    # author_kimchy = [author for author in dataset.all_authors if author.name == author_kimchy.name or author.email == author_kimchy.email]
    # dataset = DataSet([m for m in dataset.methods_with_authors if m.root_node.author in author_kimchy],author_kimchy)
    # r_index = {a: 0 for a in dataset.all_authors}
    # authors = [(0, dataset.all_authors)]
    # batches = {0: generate_batches(dataset.methods_with_authors, r_index)}
    author_egr = dataset.all_authors[:19] + dataset.all_authors[21:]
    dataset = DataSet([m for m in dataset.methods_with_authors if m.root_node.author in author_egr], author_egr)
    r_index = {a: 1 for a in dataset.all_authors}
    authors = [(1, dataset.all_authors)]
    batches = {1: generate_batches(dataset.methods_with_authors, r_index)}
    batches = prepare_batches(batches)
    _, batches = divide_data_set(batches, 0, 1000)
    gc.collect()
    res = process_batches(batches, authors, params)
    print('all data mean error {}'.format(res))


def main():
    # sys.setrecursionlimit(99999)
    # np.set_printoptions(threshold=100000)
    # gc.enable()
    # dataset: list[DataSet] = [None, None]
    #
    with open('AuthorClassifier/best_result', 'rb') as f:
        params = P.load(f)
    # with open('Dataset/CombinedProjects/author_file_compass', 'rb') as f:
    #     dataset[0] = P.load(f)  # type: DataSet
    # with open('Dataset/CombinedProjects/author_file_textmapper', 'rb') as f:
    #     dataset[1] = P.load(f)  # type: DataSet
    #
    # r_index = {a: i for i in [0, 1] for a in dataset[i].all_authors}
    # authors = [(0, dataset[0].all_authors), (1, dataset[1].all_authors)]
    #
    # batches = {0: generate_batches(dataset[0].methods_with_authors, r_index),
    #            1: generate_batches(dataset[1].methods_with_authors, r_index)}
    #
    # batches = prepare_batches(batches)
    with open('Dataset/CombinedProjects/top_5_MPS', 'rb') as f:
        dataset = P.load(f)
    dataset = dataset[:2]
    indexes = range(len(dataset))
    r_index = {aa: i for i, a in enumerate(dataset) for aa in a[1]}
    authors = [(i, dataset[i][1]) for i in indexes]
    gc.collect()
    batches = {i: generate_batches(dataset[i][0], r_index) for i in indexes}
    batches = prepare_batches(batches)
    _, batches = divide_data_set(batches, 0, 6400)
    dataset = None
    gc.collect()
    res = process_batches(batches, authors, params)
    print('all data mean error {}'.format(res))


if __name__ == '__main__':
    main()
