import numpy as np
import theano.tensor as T
from theano import function

from TBCNN.NetworkParams import NUM_FEATURES


class Layer:
    def __init__(self, bias, name="", feature_amount=NUM_FEATURES,
                 activation=T.nnet.relu,
                 is_pool=False):
        self.bias = bias

        self.name = name
        self.is_pool = is_pool

        self.feature_amount = feature_amount

        self.forward_connection = []
        self.back_connection = []

        self.activation = activation

        self.initialized = False

    def build_forward(self):
        connections = []
        for c in self.back_connection:
            connections.append(c.forward)
        if not self.is_pool:
            if self.bias is not None:
                if len(self.back_connection) == 0:
                    self.forward = function([], self.bias)
                else:
                    self.z = T.sum(connections, axis=0)
                    self.forward = function([], self.activation(T.add(self.z, self.bias)))
            else:
                self.z = T.sum(connections, axis=0)
                self.forward = function([], self.activation(self.z))
        else:
            z = connections
            # self.forward = lambda: np.max(z, axis=0)
            self.forward = function([], T.max(z, axis=0))

        self.initialized = True


class PoolLayer(Layer):
    def __init__(self, name, feature_amount):
        super().__init__(None, name, feature_amount,
                         activation=T.max, is_pool=True)
