import numpy as np
import theano.tensor as T
from theano import function
from theano import shared

from TBCNN.NetworkParams import NUM_FEATURES, BATCH_SIZE


def unknown_relu(x):
    return x * (x > 0)


class Layer:
    def __init__(self, bias, name="", feature_amount=NUM_FEATURES,
                 activation=unknown_relu,
                 is_pool=False):
        self.bias = bias

        # print("\n\n" + name + "\n")
        # if bias is not None:
        #     print(self.bias)
        #     print(self.bias.eval())
        # else:
        #     print("bias is none")

        self.name = name
        self.is_pool = is_pool

        self.feature_amount = feature_amount
        if not is_pool:
            self.z = shared(np.zeros((feature_amount, BATCH_SIZE)))
        else:
            self.z = []

        self.forward_connection = []
        self.back_connection = []

        self.activation = activation

        # self.forward = function([])

    def f_prop(self):
        self.forward()

    def build_functions(self):
        if not self.is_pool:
            if self.bias is not None:
                if len(self.back_connection) == 0:
                    self.forward = function([], self.bias)
                else:
                    self.forward = function([], self.activation(self.z + self.bias))
            else:
                self.forward = function([], self.activation(self.z))
        else:
            self.forward = lambda: np.max(self.z, axis=0)


class PoolLayer(Layer):
    def __init__(self, name, feature_amount):
        super().__init__(None, name, feature_amount,
                         activation=T.max, is_pool=True)
