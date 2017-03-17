import _pickle as P
from copy import deepcopy

from AST.Tokenizer import Author


def collapse(filename):
    with open(filename, 'rb') as f:
        dataset = P.load(f)
        authors = dataset.all_authors
    return author_collapse(authors)


def author_collapse(authors):
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
    return uauthors
