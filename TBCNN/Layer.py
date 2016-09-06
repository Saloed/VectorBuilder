import theano
import theano.compile
import theano.tensor as T
from theano.compile import SharedVariable as TS

from TBCNN.NetworkParams import NUM_FEATURES


class Layer:
    def __init__(self, bias: TS, name="", feature_amount=NUM_FEATURES,
                 activation=T.nnet.relu,
                 is_pool=False):
        self.bias = bias
        self.name = name
        self.is_pool = is_pool
        self.feature_amount = feature_amount
        self.out_connection = []
        self.in_connection = []
        self.activation = activation
        self.forward = None

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        if not self.is_pool:
            if len(connections) != 0:
                z = T.sum(connections, axis=0, acc_dtype=theano.config.floatX)
                if self.bias is not None:
                    y = self.activation(T.add(z, self.bias))
                else:
                    y = self.activation(z)
            else:
                y = self.bias
        else:
            y = T.max(connections, axis=0)
        self.forward = y


class PoolLayer(Layer):
    def __init__(self, name, feature_amount):
        super().__init__(None, name, feature_amount,
                         activation=T.max, is_pool=True)
