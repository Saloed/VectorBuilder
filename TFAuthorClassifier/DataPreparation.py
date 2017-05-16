import _pickle as P
from collections import namedtuple
from random import shuffle


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


def make_data_file(filename, min_tokens, max_child_tokens):
    with open(filename, 'rb') as f:
        data_set = P.load(f)
    data_set = list(data_set)

    # fixme: Smth strange (this token not in all_tokens but appear)
    def check_for_error(method):
        for token in method.all_nodes:
            if token.token_type == 'ERROR_ELEMENT':
                return False
        return True

    def check_max_len(method, max_len):
        def process_node(node):
            result = len(node.children) < max_len
            for child in node.children:
                if result:
                    result = process_node(child)
            return result

        return process_node(method.root_node)

    new_data_set = [
        ([m for m in methods
          if len(m.all_nodes) > min_tokens
          and check_for_error(m)
          and check_max_len(m, max_child_tokens)
          ], author) for methods, author in data_set]

    new_data_set.sort(key=lambda x: len(x[0]), reverse=True)

    with open(filename + '_{}_{}'.format(min_tokens, max_child_tokens), 'wb') as f:
        P.dump(new_data_set, f)

    return new_data_set


def main():
    with open('Dataset/TestRepos/cloudstack_file_100_100', 'rb') as f:
        # with open('TFAuthorClassifier/test_data', 'rb') as f:
        dataset = P.load(f)
    dataset = dataset[:3]
    indexes = range(len(dataset))
    authors = [(i, dataset[i][1]) for i in indexes]
    authors_amount, r_index = build_vectors(authors)
    batches = {i: dataset[i][0] for i in indexes}
    train_set, valid_set, test_set = divide_data_set(batches, 300, 100, 200)
    dataset = DataSet(test_set, valid_set, train_set, r_index, authors_amount)
    with open('Dataset/TestRepos/cloudstack_data_set', 'wb') as f:
        # with open('TFAuthorClassifier/test_data_data', 'wb') as f:
        P.dump(dataset, f)


if __name__ == '__main__':
    main()
    # generate_author_file()
