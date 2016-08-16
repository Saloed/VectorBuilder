import theano.tensor as T
from theano import function

from TBCNN.Layer import Layer


class Connection:
    def __init__(self, from_layer: Layer, to_layer: Layer,
                 weights, w_coeff=1.0):
        self.from_layer = from_layer
        self.to_layer = to_layer
        if weights is not None:
            self.weights = weights.reshape((to_layer.feature_amount, from_layer.feature_amount))
        self.w_coeff = w_coeff

        from_layer.forward_connection.append(self)
        to_layer.back_connection.append(self)

        # propagations
        if weights is not None:
            forward = T.mul(T.dot(self.weights, from_layer.forward()), self.w_coeff)
            self.forward = function([], forward, updates=[
                (to_layer.z, to_layer.z + forward)
            ])
        else:  # means pool connection
            self.forward = function([], T.concatenate([to_layer.z, from_layer.forward()]))


class PoolConnection(Connection):
    def __init__(self, from_layer, to_layer, ):
        super().__init__(from_layer, to_layer, None)
