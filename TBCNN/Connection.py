import theano.tensor as T
from theano import function
from theano.compile.function import In
from theano.compile.io import Out
from TBCNN.NetworkParams import *
import theano.printing as tp

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

    def f_prop(self):
        self.forward()

    def build_functions(self):
        # propagations

        if not self.is_pool:
            c = T.fscalar('c')
            y_f = T.fmatrix('y')

            forward = T.mul(T.dot(self.weights, y_f), c)
            self.forward = function(inputs=[
                In(c, value=self.w_coeff),
                In(y_f, value=self.from_layer.forward())
            ], outputs=forward,
                name="conn_" + self.from_layer.name + "_to_" + self.to_layer.name)

            # tp.debugprint(self.forward)

        else:  # means pool connection
            self.forward = function([], outputs=Out(self.from_layer.forward(), borrow=True))
        self.initialized = True


class PoolConnection(Connection):
    def __init__(self, from_layer, to_layer, ):
        super().__init__(from_layer, to_layer, None, is_pool=True)
