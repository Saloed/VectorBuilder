from abc import abstractmethod

import theano.tensor as T
from theano.compile import SharedVariable as TS

from TBCNN.Layer import Layer


class BaseConnection:
    def __init__(self, from_layer: Layer, to_layer: Layer):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.forward = None

        from_layer.out_connection.append(self)
        to_layer.in_connection.append(self)

    @abstractmethod
    def build_forward(self):
        pass

    def __str__(self):
        return 'con from {0} to {1}'.format(self.from_layer.name, self.to_layer.name)


class Connection(BaseConnection):
    def __init__(self, from_layer: Layer, to_layer: Layer, weights: TS, w_coeff=1.0):
        super().__init__(from_layer, to_layer)
        self.weights = weights
        self.w_coeff = w_coeff

    def build_forward(self):
        self.forward = T.mul(T.dot(self.weights, self.from_layer.forward), self.w_coeff)


class PoolConnection(BaseConnection):
    def __init__(self, from_layer: Layer, to_layer: Layer):
        super().__init__(from_layer, to_layer)

    def build_forward(self):
        self.forward = self.from_layer.forward
