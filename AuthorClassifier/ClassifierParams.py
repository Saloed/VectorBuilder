from collections import namedtuple

import theano.tensor
import numpy as np

NUM_FEATURES = 30

margin = 1
learn_rate = 0.0025  # 0.0001
beta = .0001
momentum = 0.1

NUM_CONVOLUTION = 60  # 50
NUM_DISCRIMINATIVE = 30  # 50
NUM_OUT_LAYER = 10
NUM_POOLING = 3

BATCH_SIZE = 1

RANDOM_RANGE = 0.02

NUM_RETRY = 200
NUM_EPOCH = 4000

Network = namedtuple('Network', ['forward', 'back', 'validation'])


class Params:
    def __init__(self,
                 w_left, w_right,
                 w_comb_ae, w_comb_emb,
                 w_conv_root, w_conv_left, w_conv_right,
                 w_dis_top, w_dis_left, w_dis_right,
                 w_out,
                 b_construct,
                 b_conv, b_dis, b_out, c_out, s_out,
                 embeddings):
        self.w = {
            'w_left': w_left,
            'w_right': w_right,

            'w_comb_ae': w_comb_ae,
            'w_comb_emb': w_comb_emb,

            'w_conv_root': w_conv_root,
            'w_conv_left': w_conv_left,
            'w_conv_right': w_conv_right,

            # 'w_dis_top': w_dis_top,
            # 'w_dis_left': w_dis_left,
            # 'w_dis_right': w_dis_right,

        }
        self.b = {
            'b_construct': b_construct,

            'b_conv': b_conv,
            # 'b_dis': b_dis,

        }

        self.svm = {
            'w_out': w_out,
            'b_out': b_out,
            # 'c_out': c_out,
            # 's_out': s_out

        }
        self.embeddings = embeddings