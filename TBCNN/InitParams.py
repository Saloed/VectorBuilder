import _pickle as P

import numpy as np
from theano import shared

from AST.TokenMap import token_map
from TBCNN.NetworkParams import *


def init_params(old_weights, amount=None, new_weights=None, upper=None, lower=None):
    old_len = len(old_weights)
    if new_weights is not None:
        new_weights = np.array(new_weights)
        amount = len(new_weights)
        old_weights = np.concatenate((old_weights, new_weights.reshape(-1)))
    else:
        if upper is None or lower is None:
            upper = 0.02
            lower = -.02
        tmp_weights = np.random.uniform(lower, upper, amount)
        old_weights = np.concatenate((old_weights, tmp_weights.reshape(-1)))
    return old_weights, range(old_len, old_len + amount)


def init_prepared_params():
    token_amount = len(token_map)

    pre_weights = np.array([])
    pre_biases = np.array([])

    np.random.seed(314)
    # random init weights
    pre_weights, pre_w_left_area = init_params(pre_weights, amount=num_features * num_features)
    pre_weights, pre_w_right_area = init_params(pre_weights, amount=num_features * num_features)
    # random init biases
    pre_biases, pre_b_token_area = init_params(pre_biases, amount=num_features * token_amount,
                                               upper=0.4, lower=0.6)
    pre_biases, pre_b_construct_area = init_params(pre_biases, amount=num_features)

    # load prepared params
    pre_params = P.load(open('TBCNN/preparam', 'rb'), encoding='latin1')

    pre_w = pre_params[:len(pre_weights)]
    pre_b = pre_params[len(pre_weights):]

    pre_w_left = pre_w[pre_w_left_area]
    pre_w_right = pre_w[pre_w_right_area]

    pre_b_token = pre_b[pre_b_token_area]
    pre_b_construct = pre_b[pre_b_construct_area]

    # init params
    weights = np.array([])
    biases = np.array([])

    # embeddings
    biases, b_token_area = init_params(biases, new_weights=pre_b_token)

    # left/right weights and biases (auto encoder)
    weights, w_left_area = init_params(weights, new_weights=pre_w_left)
    weights, w_right_area = init_params(weights, new_weights=pre_w_right)
    biases, b_construct_area = init_params(biases, new_weights=pre_b_construct)

    # combine embeddings and encoded
    w_ae = (np.eye(num_features) / 2).reshape(-1)
    w_emb = (np.eye(num_features) / 2).reshape(-1)

    weights, w_comb_ae_area = init_params(weights, new_weights=w_ae)
    weights, w_comb_emb_area = init_params(weights, new_weights=w_emb)

    # convolution
    weights, w_conv_root_area = init_params(weights, amount=num_features * num_convolution)
    weights, w_conv_left_area = init_params(weights, amount=num_features * num_convolution)
    weights, w_conv_right_area = init_params(weights, amount=num_features * num_convolution)
    biases, b_conv_area = init_params(biases, amount=num_convolution)

    # discriminative layer
    weights, w_dis_area = init_params(weights, amount=num_pooling * num_convolution * num_discriminative)
    biases, b_dis_area = init_params(biases, amount=num_discriminative)

    # output layer
    weights, w_out_area = init_params(weights, amount=num_discriminative * num_out_layer,
                                      upper=.0002, lower=-.0002)
    biases, b_out_area = init_params(biases, new_weights=np.zeros((num_out_layer, 1)))

    weights = weights.reshape((-1, 1))
    biases = biases.reshape((-1, 1))

    # theano wrapper
    w_left = shared(weights[w_left_area], 'W_left')
    w_right = shared(weights[w_right_area], 'W_right')

    w_comb_ae = shared(weights[w_comb_ae_area], 'W_comb_ae')
    w_comb_emb = shared(weights[w_comb_emb_area], 'W_comb_emb')

    w_conv_root = shared(weights[w_conv_root_area], 'W_conv_root')
    w_conv_left = shared(weights[w_conv_left_area], 'W_conv_left')
    w_conv_right = shared(weights[w_conv_right_area], 'W_conv_right')

    w_dis = shared(weights[w_dis_area], 'W_dis')

    w_out = shared(weights[w_out_area], 'W_out')

    b_token = shared(biases[b_token_area], 'B_token')

    b_construct = shared(biases[b_construct_area], 'B_constr')

    b_conv = shared(biases[b_conv_area], 'B_conv')

    b_dis = shared(biases[b_dis_area], 'B_dis')

    b_out = shared(biases[b_out_area], 'B_out')

    embeddings = []
    for (index, token) in enumerate(token_map):
        target = index * num_features
        area = range(target, target + num_features)
        embeddings.insert(index, shared(biases[area], 'emb_' + token))

    biases = shared(biases, 'B')
    weights = shared(weights, 'W')

    return Params(weights, biases,
                  w_left, w_right,
                  w_comb_ae, w_comb_emb,
                  w_conv_root, w_conv_left, w_conv_right,
                  w_dis, w_out,
                  b_token, b_construct,
                  b_conv, b_dis, b_out,
                  embeddings)
