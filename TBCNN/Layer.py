import numpy as np
import theano
import theano.printing as tp
import theano.tensor as T
from theano import function
from theano import shared

from TBCNN.NetworkParams import NUM_FEATURES, BATCH_SIZE

file = open('layer_activ.txt', mode='w')
fun_graph = open('function.png', mode='w')


class Layer:
    def __init__(self, bias, name="", feature_amount=NUM_FEATURES,
                 activation=T.nnet.relu,
                 is_pool=False):
        self.bias = bias

        self.name = name
        self.is_pool = is_pool

        self.feature_amount = feature_amount

        self.forward_connection = []
        self.back_connection = []

        self.activation = activation

        self.initialized = False

        def test_activ(x):
            print("\n\n", name, file=file)
            print("bias", file=file)
            if self.bias is not None:
                print(self.bias.eval(), file=file)
            else:
                print("None", file=file)
            if not is_pool:
                print("input", file=file)
                print(self.z.eval(), file=file)

            print("\n", file=file)
            print(x, file=file)
            r = T.nnet.relu(x)
            print("\n", file=file)
            print(r, file=file)
            return r

        if self.name != "softmax":
            self.activation = test_activ

    def f_prop(self):
        self.forward()

    def build_functions(self):
        connections = []
        for c in self.back_connection:
            connections.append(c.forward)
        if not self.is_pool:
            if self.bias is not None:
                if len(self.back_connection) == 0:
                    self.forward = function([], self.bias)
                else:
                    self.z = T.sum(connections, axis=0)
                    self.forward = function([], self.activation(T.add(self.z, self.bias)))
            else:
                self.z = T.sum(connections, axis=0)
                self.forward = function([], self.activation(self.z))
        else:
            z = connections
            self.forward = lambda: np.max(z, axis=0)

        self.initialized = True


class PoolLayer(Layer):
    def __init__(self, name, feature_amount):
        super().__init__(None, name, feature_amount,
                         activation=T.max, is_pool=True)
