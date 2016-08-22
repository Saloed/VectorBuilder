import theano.tensor as T

from TBCNN.Layer import Layer


class Connection:
    def __init__(self, from_layer: Layer, to_layer: Layer,
                 weights, w_coeff=1.0, is_pool=False):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.is_pool = is_pool
        self.initialized = False
        if not is_pool:
            self.weights = weights.reshape((to_layer.feature_amount, from_layer.feature_amount))
        self.w_coeff = w_coeff

        from_layer.forward_connection.append(self)
        to_layer.back_connection.append(self)

    def build_functions(self):
        # propagations

        if not self.is_pool:
            self.forward = T.mul(T.dot(self.weights, self.from_layer.forward()), self.w_coeff)
        else:  # means pool connection
            self.forward = self.from_layer.forward()
        self.initialized = True


class PoolConnection(Connection):
    def __init__(self, from_layer, to_layer, ):
        super().__init__(from_layer, to_layer, None, is_pool=True)
