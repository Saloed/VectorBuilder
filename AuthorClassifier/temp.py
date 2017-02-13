import _pickle as P
import numpy as np

from AST.GitAuthor import DataSet


def main():
    np.set_printoptions(threshold=1000000)
    with open('../Dataset/CombinedProjects/author_file_textmapper', 'rb') as f:
        dataset = P.load(f)  # type: DataSet
    print(dataset)

if __name__ == '__main__':
    main()
