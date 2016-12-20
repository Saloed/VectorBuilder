import _pickle as c_pickle

from pyclustering.cluster.syncsom import syncsom
import numpy as np


def clustering(file_with_params):
    with open(file_with_params, 'rb') as in_file:
        params = c_pickle.load(in_file)

    embs = {k: e.eval() for k, e in params.embeddings.items()}

    nearest = eps_nearest(embs, 0.4)
    for n in nearest:
        print(n)


# visited_embs = {k: False for k in embs.keys()}
#     data = list(embs.values())
#     som = syncsom(data, 30, 30, 0.9)
#     som.process()
#     clusters = som.get_clusters()
#
#     result = [[find_name(data[v], embs, visited_embs) for v in c] for c in clusters]
#
#     print('end simulation')
#     print(clusters)
#     for i, r in enumerate(result):
#         print('cluster {}'.format(i))
#         print(r)
#
#
# def find_name(value, embs, visited):
#     for k, v in embs.items():
#         if visited[k]: continue
#         if v == value:
#             visited[k] = True
#             if np.sum(v) == 0:
#                 return k + '_0'
#             return k
#     return None


def euclid_dist(x, y):
    return np.std(x - y)


class Neighbours:
    def __init__(self, label, nearest):
        self.label = label
        self.nearest = nearest
        self.nl = len(nearest)

    def __str__(self):
        res = self.label + ' : \t'
        for n in self.nearest:
            res += '{0} : {1:.3f} \t'.format(n[0], n[1])
        return res


def eps_nearest(data: dict, x):
    def find_nearest(target, data, x):
        res = [(label, euclid_dist(target, value)) for label, value in data.items()]
        res = [r for r in res if r[1] < x]
        res.sort(key=lambda t: t[1])
        return res[1:]

    result = [Neighbours(label, find_nearest(val, data, x)) for label, val in data.items()]
    result.sort(key=lambda x: x.nl)
    return result


if __name__ == '__main__':
    clustering('EmbTest/embeddings_w2v')
