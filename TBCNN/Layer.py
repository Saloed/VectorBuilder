import numpy as np
import theano.tensor as T
import theano
from theano import function
from theano import shared
import theano.compile
from TBCNN.NetworkParams import NUM_FEATURES, BATCH_SIZE


class Layer:
    def __init__(self, bias, name="", feature_amount=NUM_FEATURES,
                 activation=T.nnet.relu,
                 is_pool=False):
        self.bias = bias

        self.name = name
        self.is_pool = is_pool

        self.feature_amount = feature_amount

        self.out_connection = []
        self.in_connection = []

        self.activation = activation

        self.f_initialized = False
        self.b_initialized = False

    def build_forward(self):
        connections = []
        for c in self.in_connection:
            connections.append(c.forward)
        if not self.is_pool:
            if self.bias is not None:
                if len(self.in_connection) == 0:
                    self.y = self.bias
                else:
                    self.z = T.sum(connections, axis=0)
                    self.y = self.activation(T.add(self.z, self.bias))
            else:
                self.z = T.sum(connections, axis=0)
                self.y = self.activation(self.z)
        else:
            z = connections
            self.y = T.max(z, axis=0)

        # self.forward = function([], outputs=self.y)
        self.forward = self.y
        self.f_initialized = True

    def build_back(self):
        if not self.is_pool:
            connections = []
            for c in self.out_connection:
                connections.append(c.back())
            if len(connections) == 0:
                self.dEdY = self.forward
                self.dEdZ = self.forward
                self.back = function([], outputs=self.dEdZ)
            else:
                self.dEdY = T.sum(connections, axis=0)

                # fixme
                # this solution not works

                self.dEdZ = self.dEdY * self.forward
                if self.bias is None:
                    pass
                if len(self.in_connection) != 0:
                    self.dEdB = T.sum(self.dEdZ, axis=1)
                else:
                    self.dEdB = self.dEdY
                bias_upd = [(self.bias, self.bias + self.dEdB)]

                self.back = function([], outputs=self.dEdZ, updates=bias_upd)

        self.b_initialized = True


def cost_function():
    pass


class PoolLayer(Layer):
    def __init__(self, name, feature_amount):
        super().__init__(None, name, feature_amount,
                         activation=T.max, is_pool=True)
