from abc import abstractmethod

import theano
import theano.compile
import theano.tensor as T
from theano.compile import SharedVariable as TS

from AuthorClassifier.ClassifierParams import *


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

    def add_in_connection(self, con):
        self.in_connection.append(con)

    def add_out_coonection(self, con):
        self.out_connection.append(con)

    def __str__(self):
        return self.name


class Embedding(Layer):
    def __init__(self, emb: TS, name="emb", feature_amount=NUM_FEATURES):
        super().__init__(name, feature_amount)
        self.forward = emb

    def build_forward(self):
        pass


class Placeholder(Layer):
    def build_forward(self):
        pass

    def __init__(self, symbolic, name, feature_amount):
        super().__init__(name, feature_amount)
        self.forward = symbolic


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
                 activation=T.nnet.relu, size=0):
        super().__init__(name, feature_amount)
        self.bias = bias
        self.size = size
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


class RBF_SVM(Layer):
    def __init__(self, bias: TS, w: TS,
                 # c: TS, s: TS,
                 feature_amount, name='SVM'):
        super().__init__(name, feature_amount)
        self.b = bias
        self.w = w
        # self.c = c
        # self.s = s

    def build_forward(self):
        # assuming that only one in connection exist
        x = self.in_connection[0].forward
        # difnorm = T.std(x - self.c)
        # kernel = T.exp(difnorm / -(T.sqr(self.s)))
        # self.forward = T.dot(self.w, kernel) + self.bias
        # self.forward = T.nnet.sigmoid(T.sqrt(T.sqr(T.dot(x, self.w) + self.b).sum()))


class Pooling(Layer):
    def __init__(self, name, feature_amount=NUM_CONVOLUTION):
        super().__init__(name, feature_amount)

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        self.forward = T.max(connections, axis=0)
