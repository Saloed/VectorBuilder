import _pickle as c_pickle
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


def clustering(file_with_params):
    with open(file_with_params, 'rb') as in_file:
        params = c_pickle.load(in_file)

    embeddings = params.embeddings

    names = [e.name for e in embeddings]
    values = [e.get_value() for e in embeddings]

    X = np.array(values)
    Z = linkage(X)
    fig = plt.figure(figsize=(200, 100))
    ax = fig.add_subplot(1, 1, 1)
    dendrogram(
        Z,
        truncate_mode='level',  # show only the last p merged clusters
        p=21,  # show only the last p merged clusters
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        labels=names,
        ax=ax
    )
    fig.show()
    fig.savefig('claster.png')


if __name__ == '__main__':
    clustering('best_params')
