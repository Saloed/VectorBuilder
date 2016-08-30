import numpy as np
from Embeddings.Parameters import *
from AST.TokenMap import token_map
from theano import shared
from theano.compile import SharedVariable as SV, theano

randomizer = np.random.RandomState(314)


def init_params(old_weights, amount=None, new_weights=None, upper=None, lower=None):
    old_len = len(old_weights)
    if new_weights is not None:
        new_weights = np.array(new_weights)
        amount = len(new_weights)
        old_weights = np.concatenate((old_weights, new_weights.reshape(-1)))
    else:
        if upper is None or lower is None:
            upper = -.002
            lower = 0.002
        tmp_weights = randomizer.uniform(lower, upper, amount)
        old_weights = np.concatenate((old_weights, tmp_weights.reshape(-1)))
    return old_weights, range(old_len, old_len + amount)


def initialize():
    token_amount = len(token_map)

    weights = np.array([])
    biases = np.array([])

    weights, w_left_range = init_params(weights, NUM_FEATURES * NUM_FEATURES)
    weights, w_right_range = init_params(weights, NUM_FEATURES * NUM_FEATURES)
    biases, _ = init_params(biases, NUM_FEATURES * token_amount)

    w_left = shared(np.asarray(weights[w_left_range].reshape((NUM_FEATURES, NUM_FEATURES)),
                               dtype=theano.config.floatX), 'w_left')
    w_right = shared(np.asarray(weights[w_right_range].reshape((NUM_FEATURES, NUM_FEATURES))
                                , dtype=theano.config.floatX), 'w_right')

    embeddings = [SV] * token_amount
    for token, index in token_map.items():
        target = index * NUM_FEATURES
        area = range(target, target + NUM_FEATURES)
        emb = np.asarray(biases[area], dtype=theano.config.floatX)
        embeddings[index] = shared(emb, 'emb_' + token)

    return Parameters(w_left, w_right, embeddings)
