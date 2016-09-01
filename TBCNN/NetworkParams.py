import theano.tensor
import numpy as np

from AST.TokenMap import token_map

NUM_FEATURES = 30

margin = 1
learn_rate = 0.0025  # 0.0001
beta = .0001
momentum = 0.1

NUM_CONVOLUTION = 600  # 50
NUM_DISCRIMINATIVE = 600  # 50
NUM_OUT_LAYER = 104
NUM_POOLING = 3

# test parameter
DONT_MAKE_CONV = False

BATCH_SIZE = 1


class Updates:
    def __init__(self):
        self.bias_updates = dict()
        self.weights_updates = dict()

        self.error = theano.tensor.fvector('E')

        def zeros(size):
            return theano.shared(np.zeros(size, dtype=theano.config.floatX))

        self.grad_w = {
            'w_left': zeros((NUM_FEATURES, NUM_FEATURES)),
            'w_right': zeros((NUM_FEATURES, NUM_FEATURES)),

            'w_comb_ae': zeros((NUM_FEATURES, NUM_FEATURES)),
            'w_comb_emb': zeros((NUM_FEATURES, NUM_FEATURES)),

            'w_conv_root': zeros((NUM_CONVOLUTION, NUM_FEATURES)),
            'w_conv_left': zeros((NUM_CONVOLUTION, NUM_FEATURES)),
            'w_conv_right': zeros((NUM_CONVOLUTION, NUM_FEATURES)),

            'w_dis_top': zeros((NUM_DISCRIMINATIVE, NUM_CONVOLUTION)),
            'w_dis_left': zeros((NUM_DISCRIMINATIVE, NUM_CONVOLUTION)),
            'w_dis_right': zeros((NUM_DISCRIMINATIVE, NUM_CONVOLUTION)),

            'w_out': zeros((NUM_OUT_LAYER, NUM_DISCRIMINATIVE))
        }
        self.grad_b = {
            'b_token': zeros(NUM_FEATURES),
            'b_construct': zeros(NUM_FEATURES),

            'b_conv': zeros(NUM_CONVOLUTION),
            'b_dis': zeros(NUM_DISCRIMINATIVE),
            'b_out': zeros(NUM_OUT_LAYER)
        }
        for token, index in token_map.items():
            self.grad_b['emb_' + token] = zeros(NUM_FEATURES)


class Network:
    def __init__(self):
        # self.layers = layers
        self.forward = None
        self.back = None


class Params:
    def __init__(self,
                 w_left, w_right,
                 w_comb_ae, w_comb_emb,
                 w_conv_root, w_conv_left, w_conv_right,
                 w_dis_top, w_dis_left, w_dis_right,
                 w_out,
                 b_token, b_construct,
                 b_conv, b_dis, b_out,
                 embeddings):
        self.w = {
            'w_left': w_left,
            'w_right': w_right,

            'w_comb_ae': w_comb_ae,
            'w_comb_emb': w_comb_emb,

            'w_conv_root': w_conv_root,
            'w_conv_left': w_conv_left,
            'w_conv_right': w_conv_right,

            'w_dis_top': w_dis_top,
            'w_dis_left': w_dis_left,
            'w_dis_right': w_dis_right,

            'w_out': w_out
        }
        self.b = {
            'b_token': b_token,
            'b_construct': b_construct,

            'b_conv': b_conv,
            'b_dis': b_dis,
            'b_out': b_out
        }
        self.embeddings = embeddings
