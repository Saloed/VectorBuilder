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
from Utils.ConfMatrix import ConfMatrix


def process_batches(batches, authors, nparams, cm):
    author_amount, r_index = build_vectors(authors)
    error = 0
    size = len(batches)
    for i, batch in enumerate(batches):
        # print('{}/{}'.format(i,size))
        author = r_index[batch.author]
        net = construct_from_nodes(batch.ast, nparams, BuildMode.train, author, author_amount)  # type: NetLoss
        fun = function([], [net.net_forward, net.error])
        frwd, err = fun()
        frwd = np.asarray(frwd[0])
        res = np.argmax(frwd)
        cm.add(res, author[1])
        print('err {0} res {1} tar {2} total {3} frwd {4}'.format(err, res, author, error, frwd))
        error += err
    return error / size


def main():
    sys.setrecursionlimit(99999)
    np.set_printoptions(threshold=100, precision=3, suppress=True)
    gc.enable()

    with open('AuthorClassifier/res_16.02.2017', 'rb') as f:
        params = P.load(f)
    with open('Dataset/CombinedProjects/top_authors_MPS', 'rb') as f:
        dataset = P.load(f)

    dataset = dataset[:5]
    indexes = range(len(dataset))
    r_index = {aa: i for i, a in enumerate(dataset) for aa in a[1]}
    authors = [(i, dataset[i][1]) for i in indexes]
    gc.collect()
    batches = {i: generate_batches(dataset[i][0], r_index) for i in indexes}
    batches = prepare_batches(batches)
    _, test_set = divide_data_set(batches, 40, 1000)
    dataset, batches = (None, None)
    gc.collect()
    cm = ConfMatrix(5)
    res = process_batches(test_set, authors, params, cm)
    cm.calc()
    print(str(cm))
    cm.print_conf_matrix()
    print('all data mean error {}'.format(res))


if __name__ == '__main__':
    main()
