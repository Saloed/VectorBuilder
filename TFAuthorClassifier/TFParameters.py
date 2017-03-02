import _pickle as P
import tensorflow as tf
from AuthorClassifier.ClassifierParams import *
import numpy as np


class Params:
    def __init__(self,
                 w_conv_root, w_conv_left, w_conv_right,
                 w_hid, w_out,
                 b_conv, b_hid, b_out,
                 embeddings, emb_values):
        self.w = {
            'w_conv_root': w_conv_root,
            'w_conv_left': w_conv_left,
            'w_conv_right': w_conv_right,
            'w_hid': w_hid,
            'w_out': w_out,
        }
        self.b = {
            'b_conv': b_conv,
            'b_hid': b_hid,
            'b_out': b_out
        }

        self.embeddings = embeddings
        self.emb_values = emb_values


def rand_weight(shape_0, shape_1, name):
    return tf.Variable(
        tf.truncated_normal(shape=[shape_0, shape_1], stddev=RANDOM_RANGE),
        name=name)


def rand_bias(shape, name):
    return tf.Variable(
        tf.truncated_normal(shape=[shape, 1], stddev=RANDOM_RANGE),
        name=name)


def init_params(author_amount):
    with open('TFAuthorClassifier/embeddings', 'rb') as f:
        np_embs = P.load(f)
    with tf.name_scope('Embeddings'):
        embeddings = {name: tf.placeholder(tf.float32, [NUM_FEATURES, 1], name) for name in np_embs.keys()}
        emb_values = {embeddings[name]: np.expand_dims(val, 1) for name, val in np_embs.items()}
    with tf.name_scope('Params'):
        w_conv_root = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_root')
        w_conv_left = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_left')
        w_conv_right = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_right')
        w_hid = rand_weight(NUM_HIDDEN, NUM_CONVOLUTION, 'w_hid')
        w_out = rand_weight(author_amount, NUM_HIDDEN, 'w_out')

        b_conv = rand_bias(NUM_CONVOLUTION, 'b_conv')
        b_hid = rand_bias(NUM_HIDDEN, 'b_hid')
        b_out = rand_bias(author_amount, 'b_out')

    return Params(w_conv_root, w_conv_left, w_conv_right,
                  w_hid, w_out, b_conv, b_hid, b_out,
                  embeddings, emb_values)
