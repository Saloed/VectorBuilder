import theano.tensor as T
from theano import function
from TBCNN.Layer import Layer
from TBCNN.NetworkParams import Updates


class Connection:
    def __init__(self, from_layer: Layer, to_layer: Layer,
                 weights, w_coeff=1.0, is_pool=False):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.is_pool = is_pool
        self.f_initialized = False
        self.b_initialized = False
        self.weights = weights
        self.w_coeff = w_coeff

        from_layer.out_connection.append(self)
        to_layer.in_connection.append(self)

    def build_forward(self):
        if not self.is_pool:
            self.forward = T.mul(T.dot(self.weights, self.from_layer.forward), self.w_coeff)
        else:
            self.forward = self.from_layer.forward
        self.f_initialized = True

    def build_back(self, updates: Updates):
        if not self.is_pool:
            dEdZ = self.to_layer.back
            dEdX = T.dot(self.weights.T, dEdZ) * self.w_coeff
            dEdW = T.dot(dEdZ, self.from_layer.forward.T) * self.w_coeff
            upd = updates.weights_updates.get(self.weights, None)
            if upd is not None:
                updates.weights_updates[self.weights] = upd + dEdW
            else:
                updates.weights_updates[self.weights] = dEdW
            self.back = dEdX
        else:
            self.back = self.to_layer.back
        self.b_initialized = True


class PoolConnection(Connection):
    def __init__(self, from_layer, to_layer, ):
        super().__init__(from_layer, to_layer, None, is_pool=True)
