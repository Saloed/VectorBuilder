from abc import abstractmethod
import tensorflow as tf

from NN.TFLayer import Layer


class BaseConnection:
    def __init__(self, from_layer: Layer, to_layer: Layer):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.forward = None

        from_layer.add_out_connection(self)
        to_layer.add_in_connection(self)

    @abstractmethod
    def build_forward(self):
        pass

    def __str__(self):
        return 'con from {0} to {1}'.format(self.from_layer.name, self.to_layer.name)


class Connection(BaseConnection):
    def __init__(self, from_layer: Layer, to_layer: Layer, weights: tf.Variable, w_coeff=1.0):
        super().__init__(from_layer, to_layer)
        self.weights = weights
        self.w_coeff = w_coeff

    def build_forward(self):
        self.forward = tf.multiply(tf.matmul(self.from_layer.forward, self.weights), self.w_coeff)


class PoolConnection(BaseConnection):
    def __init__(self, from_layer: Layer, to_layer: Layer):
        super().__init__(from_layer, to_layer)

    def build_forward(self):
        self.forward = self.from_layer.forward
