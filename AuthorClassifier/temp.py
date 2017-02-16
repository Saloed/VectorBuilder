import _pickle as P
import numpy as np
from copy import deepcopy


class Author:
    def __init__(self, author):
        self.name = author.name
        self.email = author.email
        self.author = author

    def __eq__(self, other):
        return self.name == other.name or self.email == other.email

    def __str__(self):
        return '{} <{}>'.format(self.name, self.email)

    def __repr__(self):
        return '{} <{}>'.format(self.name, self.email)


def main():
    np.set_printoptions(threshold=1000000)
    with open('Dataset/CombinedProjects/author_file_MPS', 'rb') as f:
        dataset = P.load(f)
        authors = dataset.all_authors
    authors = [Author(a) for a in authors]
    acopy = deepcopy(authors)
    uauthors = []

    while len(acopy) > 0:
        change = True
        tmp = [acopy[0]]
        del acopy[0]
        while change:
            change = False
            for i in range(len(acopy)):
                if acopy[i] in tmp:
                    tmp.append(acopy[i])
                    acopy[i] = None
                    change = True
            acopy = [a for a in acopy if a is not None]
        uauthors.append(tmp)
    return authors


if __name__ == '__main__':
    main()
