import numpy as np
from Embeddings.Parameters import *
from theano import shared
from theano.compile import SharedVariable as SV, theano

randomizer = np.random.RandomState(314)


def reshape_and_cast(param, shape, dtype):
    return np.asarray(param.reshape(shape),
                      dtype=dtype)


def rand_param(shape_0, shape_1=1, name='', dtype=theano.config.floatX):
    size = shape_0 * shape_1
    param = randomizer.uniform(-RANDOM_RANGE, RANDOM_RANGE, size)
    if shape_1 != 1:
        param = reshape_and_cast(param, (shape_0, shape_1), dtype)
    else:
        param = reshape_and_cast(param, shape_0, dtype)
    return shared(param, name=name)


def initialize(tokens):
    dtype = theano.config.floatX
    w_left = rand_param(NUM_FEATURES, NUM_FEATURES, 'w_left', dtype)
    w_right = rand_param(NUM_FEATURES, NUM_FEATURES, 'w_right', dtype)
    b_construct = rand_param(NUM_FEATURES, name='b_construct', dtype=dtype)

    embeddings = {}
    for token in tokens:
        embeddings[token] = rand_param(NUM_FEATURES, name='emb_' + token, dtype=dtype)

    return Parameters(w_left, w_right, b_construct, embeddings)
