import numpy as np
import theano
import _pickle as P
from numpy.random import RandomState
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


def init_params(all_authors):
    with open('emb_params', 'rb') as p_file:
        e_params = P.load(p_file)
    out_features = len(all_authors)
    embeddings = e_params.embeddings

    b_construct = e_params.b_construct

    w_left = e_params.w['w_left']
    w_right = e_params.w['w_right']

    diag_matrix = (np.eye(NUM_FEATURES) / 2).reshape((NUM_FEATURES, NUM_FEATURES))

    w_comb_ae = shared(np.asarray(diag_matrix, dtype=theano.config.floatX), 'w_comb_ae')
    w_comb_emb = shared(np.asarray(diag_matrix, dtype=theano.config.floatX), 'w_comb_emb')
    # w_comb_ae = rand_weight(NUM_FEATURES, NUM_FEATURES, 'w_comb_ae')
    # w_comb_emb = rand_weight(NUM_FEATURES, NUM_FEATURES, 'w_comb_emb')

    w_conv_root = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_root')
    w_conv_left = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_left')
    w_conv_right = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_right')

    w_dis_top = rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_top')
    w_dis_left = rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_left')
    w_dis_right = rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_right')

    w_out = rand_weight(out_features, NUM_DISCRIMINATIVE, 'w_out')

    b_conv = rand_bias(NUM_CONVOLUTION, 'b_conv')

    b_dis = rand_bias(NUM_DISCRIMINATIVE, 'b_dis')

    b_out = rand_bias(out_features, 'b_out')

    return Params(w_left, w_right,
                  w_comb_ae, w_comb_emb,
                  w_conv_root, w_conv_left, w_conv_right,
                  w_dis_top, w_dis_left, w_dis_right,
                  w_out,
                  b_construct,
                  b_conv, b_dis, b_out,
                  embeddings)
