import numpy as np
import theano.tensor as T
from theano import function
from theano import shared

from TBCNN.NetworkParams import NUM_FEATURES, BATCH_SIZE


class Layer:
    def __init__(self, bias, name="", feature_amount=NUM_FEATURES,
                 activation=T.tanh, is_pool=False):
        self.bias = bias

        self.name = name

        self.feature_amount = feature_amount
        if not is_pool:
            self.z = shared(np.zeros((feature_amount, BATCH_SIZE)))
        else:
            self.z = []

        self.forward_connection = []
        self.back_connection = []

        self.activation = activation
        if not is_pool:
            if bias is not None:
                self.forward = function([], self.activation(self.z + self.bias), updates=[
                    (self.z, self.z + bias)
                ])
            else:
                self.forward = function([], self.activation(self.z))
        else:

            self.forward = lambda: np.max(self.z, axis=0)

    def f_prop(self):
        self.forward()


class PoolLayer(Layer):
    def __init__(self, name, feature_amount):
        super().__init__(None, name, feature_amount,
                         activation=T.max, is_pool=True)
