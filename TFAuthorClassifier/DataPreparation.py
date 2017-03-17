import _pickle as P
from collections import namedtuple
from random import shuffle

from AST.GitAuthor import get_repo_methods_with_authors


def generate_author_file():
    data_set = get_repo_methods_with_authors('../Dataset/intellij-community/')
    # data_set = get_single_author_data('../Dataset/OneAuthorProjects/distributedlog/')
    with open('../Dataset/author_file_intellij', 'wb') as f:
        P.dump(data_set, f)


def divide_data_set(data_set, train_units, valid_units, test_units):
    data_set = list(zip(*data_set.values()))
    if train_units + valid_units + test_units > len(data_set):
        raise Exception('Too much units to divide')
    shuffle(data_set)
    train_set = [d for data in data_set[:train_units] for d in data]
    v_end = train_units + valid_units
    valid_set = [d for data in data_set[train_units:v_end] for d in data]
    test_set = [d for data in data_set[v_end:v_end + test_units] for d in data]
    return train_set, valid_set, test_set


def build_vectors(authors):
    size = len(authors)
    index = {author: uauthor[0] for uauthor in authors for author in uauthor[1]}
    return size, index


DataSet = namedtuple('DataSet', ['test', 'valid', 'train', 'r_index', 'amount'])


def main():
    with open('Dataset/intellij_data', 'rb') as f:
        # with open('TFAuthorClassifier/test_data', 'rb') as f:
        dataset = P.load(f)
    dataset = dataset[:5]
    indexes = range(len(dataset))
    authors = [(i, dataset[i][1]) for i in indexes]
    authors_amount, r_index = build_vectors(authors)

    # fixme: Smth strange (this token not in all_tokens but appear)
    def check_for_error(method):
        for token in method.all_nodes:
            if token.token_type == 'ERROR_ELEMENT':
                return False
        return True

    batches = {i: [m for m in dataset[i][0] if check_for_error(m)] for i in indexes}
    train_set, valid_set, test_set = divide_data_set(batches, 900, 200, 700)
    dataset = DataSet(test_set, valid_set, train_set, r_index, authors_amount)
    with open('Dataset/intellij_data_set', 'wb') as f:
        # with open('TFAuthorClassifier/test_data_data', 'wb') as f:
        P.dump(dataset, f)


if __name__ == '__main__':
    main()
    # generate_author_file()
