from abc import abstractmethod

import theano
import theano.compile
import theano.tensor as T
from theano.compile import SharedVariable as TS

from TBCNN.NetworkParams import *


class Layer:
    def __init__(self, name, feature_amount):
        self.name = name
        self.feature_amount = feature_amount
        self.out_connection = []
        self.in_connection = []
        self.forward = None

    @abstractmethod
    def build_forward(self):
        pass

    def __str__(self):
        return self.name


class Embedding(Layer):
    def __init__(self, emb: TS, name="emb", feature_amount=NUM_FEATURES):
        super().__init__(name, feature_amount)
        self.forward = emb

    def build_forward(self):
        pass


class Combination(Layer):
    def __init__(self, name="comb", feature_amount=NUM_FEATURES):
        super().__init__(name, feature_amount)

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        self.forward = T.sum(connections, axis=0, acc_dtype=theano.config.floatX)


class Encoder(Layer):
    def __init__(self, bias: TS, name="encode", feature_amount=NUM_FEATURES,
                 activation=T.tanh):
        super().__init__(name, feature_amount)
        self.bias = bias
        self.activation = activation

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        z = T.sum(connections, axis=0, acc_dtype=theano.config.floatX)
        self.forward = self.activation(T.add(z, self.bias))


class Convolution(Layer):
    def __init__(self, bias: TS, name="conv", feature_amount=NUM_CONVOLUTION,
                 activation=T.nnet.relu):
        super().__init__(name, feature_amount)
        self.bias = bias
        self.activation = activation

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        z = T.sum(connections, axis=0, acc_dtype=theano.config.floatX)
        self.forward = self.activation(T.add(z, self.bias))


class FullConnected(Layer):
    def __init__(self, bias: TS,
                 activation, name="fc", feature_amount=NUM_DISCRIMINATIVE):
        super().__init__(name, feature_amount)
        self.bias = bias
        self.activation = activation

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        z = T.sum(connections, axis=0, acc_dtype=theano.config.floatX)
        self.forward = self.activation(T.add(z, self.bias))


class Pooling(Layer):
    def __init__(self, name, feature_amount=NUM_CONVOLUTION):
        super().__init__(name, feature_amount)

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        self.forward = T.max(connections, axis=0)
