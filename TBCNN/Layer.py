import numpy as np
import theano.tensor as T
from theano import function
from theano import shared

from TBCNN.NetworkParams import NUM_FEATURES, NUM_DATA


class Layer:
    def __init__(self, bias, name="", feature_amount=NUM_FEATURES,
                 activation=T.tanh):
        self.bias = bias

        self.name = name

        self.feature_amount = feature_amount

        self.z = shared(np.zeros((feature_amount, NUM_DATA)))

        self.forward_connection = []
        self.back_connection = []

        self.activation = activation
        if bias is not None:
            self.forward = function([], self.activation(self.z + self.bias), updates=[
                (self.z, self.z + bias)
            ])
        else:
            self.forward = function([], self.activation(self.z))


class PoolLayer(Layer):
    def __init__(self, name, feature_amount):
        super().__init__(None, name, feature_amount,
                         activation=T.max)
