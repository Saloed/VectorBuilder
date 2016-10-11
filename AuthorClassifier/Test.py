import sys
import _pickle as P

from AuthorClassifier.Builder import construct_from_nodes
from AuthorClassifier.Train import collapse_authors, generate_batches, build_vectors
from AuthorClassifier.Builder import BuildMode


def process_batches(batches, authors, params):
    author_amount = len(authors)
    reverse_index = build_vectors(authors)
    for i, batch in enumerate(batches):
        author = reverse_index[batch.author]
        print('build {}'.format(i))
        net = construct_from_nodes(batch.ast, params, BuildMode.test, author_amount)
        result = net()
        print(batch.author, author, result)


def main():
    with open('../Dataset/author_file', 'rb') as f:
        dataset = P.load(f)

    with open('classifier_params', 'rb') as f:
        params = P.load(f)

    sys.setrecursionlimit(99999)

    all_authors = dataset.all_authors
    authors = collapse_authors(all_authors)
    all_batches = generate_batches(dataset.methods_with_authors)

    process_batches(all_batches, authors, params)


if __name__ == '__main__':
    main()
