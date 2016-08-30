import numpy as np
import theano.tensor as T
import theano
from theano import function
from theano import shared
import theano.compile
from TBCNN.NetworkParams import Updates
from TBCNN.NetworkParams import NUM_FEATURES, BATCH_SIZE
from theano.compile import SharedVariable as TS


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
        self.f_initialized = False
        self.b_initialized = False

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        if not self.is_pool:
            if self.bias is not None:
                if len(self.in_connection) == 0:
                    self.y = self.bias
                else:
                    self.z = T.sum(connections, axis=0)
                    self.y = self.activation(T.add(self.z, self.bias))
            else:
                self.z = T.sum(connections, axis=0)
                self.y = self.activation(self.z)
        else:
            self.y = T.max(connections, axis=0)
        self.forward = self.y
        self.f_initialized = True

    def build_back(self, updates: Updates):
        if not self.is_pool:
            connections = [c.back for c in self.out_connection]
            if len(connections) == 0:
                dEdY = updates.error
                dEdZ = updates.error
                self.back = dEdZ
            else:
                dEdY = T.sum(connections, axis=0)
                dEdZ = dEdY * np.array(self.forward != 0,
                                       dtype=theano.config.floatX)
                if self.bias is None:
                    self.back = dEdZ
                else:
                    if len(self.in_connection) == 0:
                        dEdB = dEdY
                    else:
                        dEdB = T.sum(dEdZ, axis=1)
                    bias_upd = dEdB.reshape((-1, 1))
                    upd = updates.bias_updates.get(self.bias.name, None)
                    if upd is not None:
                        updates.bias_updates[self.bias.name] = upd + bias_upd
                    else:
                        updates.bias_updates[self.bias.name] = bias_upd
                    self.back = dEdZ
        else:
            self.back = self.out_connection[0].back
        self.b_initialized = True


class PoolLayer(Layer):
    def __init__(self, name, feature_amount):
        super().__init__(None, name, feature_amount,
                         activation=T.max, is_pool=True)
