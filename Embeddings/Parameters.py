from theano.compile import SharedVariable as SV, theano
import numpy as np

from AST.TokenMap import token_map

NUM_FEATURES = 30
MARGIN = 1
LEARN_RATE = 0.025
MOMENTUM = 0.1


class Parameters:
    def __init__(self,
                 w_left: SV, w_right: SV, embeddings: list):
        self.w = {
            'w_left': w_left,
            'w_right': w_right
        }

        self.embeddings = embeddings


class Updates:
    def __init__(self):
        # self.bias_updates = dict()
        # self.weights_updates = dict()

        self.target = theano.tensor.fvector('Target')

        # def zeros(size):
        #     return theano.shared(np.zeros(size, dtype=theano.config.floatX))
        #
        # self.grad_b = dict()
        # for token, index in token_map.items():
        #     self.grad_b['emb_' + token] = zeros(NUM_FEATURES)
        #
        # self.grad_w = {
        #     'w_left': zeros((NUM_FEATURES, NUM_FEATURES)),
        #     'w_right': zeros((NUM_FEATURES, NUM_FEATURES)),
        # }
