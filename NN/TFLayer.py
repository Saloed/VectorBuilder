from abc import abstractmethod
import tensorflow as tf

from AuthorClassifier.ClassifierParams import NUM_FEATURES, NUM_CONVOLUTION, NUM_DISCRIMINATIVE


class Layer:
    def __init__(self, name, feature_amount):
        self.name = name
        self.feature_amount = feature_amount
        self.out_connection = []
        self.in_connection = []
        self.forward = None

    @abstractmethod
    def build_forward(self):
        pass

    def add_in_connection(self, con):
        self.in_connection.append(con)

    def add_out_coonection(self, con):
        self.out_connection.append(con)

    def __str__(self):
        return self.name


class Embedding(Layer):
    def __init__(self, emb, name="emb", feature_amount=NUM_FEATURES):
        super().__init__(name, feature_amount)
        self.forward = emb

    def build_forward(self):
        pass


class Placeholder(Layer):
    def build_forward(self):
        pass

    def __init__(self, symbolic, name, feature_amount):
        super().__init__(name, feature_amount)
        self.forward = symbolic


class Combination(Layer):
    def __init__(self, name="comb", feature_amount=NUM_FEATURES):
        super().__init__(name, feature_amount)

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        self.forward = tf.reduce_sum(connections, axis=0)


class Encoder(Layer):
    def __init__(self, bias: tf.Variable, name="encode", feature_amount=NUM_FEATURES,
                 activation=tf.tanh):
        super().__init__(name, feature_amount)
        self.bias = bias
        self.activation = activation

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        z = tf.reduce_sum(connections, axis=0)
        self.forward = self.activation(tf.add(z, self.bias))


class Convolution(Layer):
    def __init__(self, bias: tf.Variable, name="conv", feature_amount=NUM_CONVOLUTION,
                 activation=tf.nn.relu, size=0):
        super().__init__(name, feature_amount)
        self.bias = bias
        self.size = size
        self.activation = activation

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        z = tf.reduce_sum(connections, axis=0)
        self.forward = self.activation(tf.add(z, self.bias))


class FullConnected(Layer):
    def __init__(self, bias: tf.Variable,
                 activation, name="fc", feature_amount=NUM_DISCRIMINATIVE):
        super().__init__(name, feature_amount)
        self.bias = bias
        self.activation = activation

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        z = tf.reduce_sum(connections, axis=0)
        self.forward = self.activation(tf.add(z, self.bias))


class Pooling(Layer):
    def __init__(self, name, feature_amount=NUM_CONVOLUTION):
        super().__init__(name, feature_amount)

    def build_forward(self):
        connections = [c.forward for c in self.in_connection]
        self.forward = tf.reduce_max(connections, axis=0)
