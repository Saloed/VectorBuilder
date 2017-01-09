import _pickle as P

import numpy as np
import theano
from numpy.random.mtrand import RandomState
from theano import shared

from AuthorClassifier.ClassifierParams import *

randomizer = RandomState(314)


def rand_weight(shape_0, shape_1, name):
    weight = randomizer.uniform(-RANDOM_RANGE, RANDOM_RANGE, shape_0 * shape_1)
    weight = np.asarray(weight.reshape((shape_0, shape_1)), dtype=theano.config.floatX)
    return shared(weight, name)


def rand_bias(shape, name):
    bias = randomizer.uniform(-RANDOM_RANGE, RANDOM_RANGE, shape)
    bias = np.asarray(bias.reshape(shape), dtype=theano.config.floatX)
    return shared(bias, name)


def init_params(all_authors, emb_path):
    with open(emb_path, 'rb') as p_file:
        e_params = P.load(p_file)
    out_features = len(all_authors)
    embeddings = e_params.embeddings

    w_conv_root = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_root')
    w_conv_left = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_left')
    w_conv_right = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_right')
    w_hid = rand_weight(NUM_HIDDEN, NUM_CONVOLUTION, 'w_hid')
    w_out = rand_weight(1, NUM_HIDDEN, 'w_out')

    b_conv = rand_bias(NUM_CONVOLUTION, 'b_conv')
    b_hid = rand_bias(NUM_HIDDEN, 'b_hid')
    b_out = rand_bias(1, 'b_out')

    return Params(w_conv_root, w_conv_left, w_conv_right,
                  w_hid, w_out, b_conv, b_hid, b_out,
                  embeddings)
