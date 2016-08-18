import theano.tensor as T
from theano import function

from TBCNN.Layer import Layer


class Connection:
    def __init__(self, from_layer: Layer, to_layer: Layer,
                 weights, w_coeff=1.0, is_pool=False):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.is_pool = is_pool
        if not is_pool:
            self.weights = weights.reshape((to_layer.feature_amount, from_layer.feature_amount))
        self.w_coeff = w_coeff

        from_layer.forward_connection.append(self)
        to_layer.back_connection.append(self)

        # self.forward = function([])

    def f_prop(self):
        self.forward()

    def build_functions(self):
        # propagations

        if not self.is_pool:
            forward = T.mul(T.dot(self.weights, self.from_layer.forward()), self.w_coeff)
            self.forward = function([], forward, updates=[
                (self.to_layer.z, self.to_layer.z + forward)
            ])
        else:  # means pool connection
            self.forward = function([], self.to_layer.z.append(self.from_layer.forward()))


class PoolConnection(Connection):
    def __init__(self, from_layer, to_layer, ):
        super().__init__(from_layer, to_layer, None, is_pool=True)
