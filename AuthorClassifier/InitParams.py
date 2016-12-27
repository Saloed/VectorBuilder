import numpy as np
import theano
import _pickle as P

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


def reset_params(params: Params, all_authors, emb_path):
    out_features = len(all_authors)

    with open(emb_path, 'rb') as p_file:
        e_params = P.load(p_file)
    out_features = len(all_authors)
    embeddings = e_params.embeddings

    b_construct = e_params.b_construct

    w_left = e_params.w['w_left']
    w_right = e_params.w['w_right']

    diag_matrix = (np.eye(NUM_FEATURES) / 2).reshape((NUM_FEATURES, NUM_FEATURES))
    diag_matrix = np.asarray(diag_matrix, dtype=theano.config.floatX)
    params.w['w_comb_ae'].set_value(diag_matrix)
    params.w['w_comb_emb'].set_value(diag_matrix)

    params.w['w_conv_root'].set_value(rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_root').get_value())
    params.w['w_conv_left'].set_value(rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_left').get_value())
    params.w['w_conv_right'].set_value(rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_right').get_value())

    # params.w['w_dis_top'].set_value(rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_top').get_value())
    # params.w['w_dis_left'].set_value(rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_left').get_value())
    # params.w['w_dis_right'].set_value(rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_right').get_value())

    params.svm['w_out'].set_value(rand_weight(out_features, NUM_CONVOLUTION, 'w_out').get_value())

    params.b['b_conv'].set_value(rand_bias(NUM_CONVOLUTION, 'b_conv').get_value())
    # params.b['b_dis'].set_value(rand_bias(NUM_DISCRIMINATIVE, 'b_dis').get_value())

    params.svm['b_out'].set_value(rand_bias(out_features, 'b_out').get_value())
    # params.svm['c_out'].set_value(rand_bias(NUM_CONVOLUTION, 'c_out').get_value())
    # params.svm['s_out'].set_value(rand_bias(NUM_CONVOLUTION, 's_out').get_value())

    return params


def init_params(all_authors, emb_path):
    with open(emb_path, 'rb') as p_file:
        e_params = P.load(p_file)
    out_features = len(all_authors)
    embeddings = e_params.embeddings

    # b_construct = e_params.b_construct
    #
    # w_left = e_params.w['w_left']
    # w_right = e_params.w['w_right']

    # w_left = rand_weight(NUM_FEATURES, NUM_FEATURES, 'w_left')
    # w_right = rand_weight(NUM_FEATURES, NUM_FEATURES, 'w_right')

    # b_construct = rand_bias(NUM_FEATURES, 'b_construct')

    # diag_matrix = (np.eye(NUM_FEATURES) / 2).reshape((NUM_FEATURES, NUM_FEATURES))

    # w_comb_ae = shared(np.asarray(diag_matrix, dtype=theano.config.floatX), 'w_comb_ae')
    # w_comb_emb = shared(np.asarray(diag_matrix, dtype=theano.config.floatX), 'w_comb_emb')
    # w_comb_ae = rand_weight(NUM_FEATURES, NUM_FEATURES, 'w_comb_ae')
    # w_comb_emb = rand_weight(NUM_FEATURES, NUM_FEATURES, 'w_comb_emb')

    w_conv_root = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_root')
    w_conv_left = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_left')
    w_conv_right = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_right')

    # w_dis_top = rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_top')
    # w_dis_left = rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_left')
    # w_dis_right = rand_weight(NUM_DISCRIMINATIVE, NUM_CONVOLUTION, 'w_dis_right')

    w_out = rand_weight(1, NUM_CONVOLUTION, 'w_out')

    b_conv = rand_bias(NUM_CONVOLUTION, 'b_conv')

    # b_dis = rand_bias(NUM_DISCRIMINATIVE, 'b_dis')
    #
    b_out = rand_bias(1, 'b_out')
    # c_out = rand_bias(NUM_CONVOLUTION, 'c_out')
    # s_out = rand_bias(NUM_CONVOLUTION, 's_out')

    return Params(None, None, None, None,
                  w_conv_root, w_conv_left, w_conv_right,
                  None, None, None,
                  w_out, None, b_conv, None, b_out,
                  None, None, embeddings)

    # return Params(w_left, w_right,
    #               w_comb_ae, w_comb_emb,
    #               w_conv_root, w_conv_left, w_conv_right,
    #               w_dis_top, w_dis_left, w_dis_right,
    #               w_out,
    #               b_construct,
    #               b_conv, b_dis, b_out, c_out, s_out,
    #               embeddings)
