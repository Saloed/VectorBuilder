import _pickle as P
from collections import OrderedDict

import numpy as np
import tensorflow as tf

LEARN_RATE = 0.07  # 0.001
L2_PARAM = 5.e-5
NUM_FEATURES = 100
NUM_CONVOLUTION = 800  # 50
NUM_HIDDEN = 80
SAVE_PERIOD = 20
BATCH_SIZE = 20
NUM_RETRY = 200
TOKEN_THRESHOLD = 150
RANDOM_RANGE = 0.02
NUM_EPOCH = 50


class Params:
    def __init__(self,
                 w_conv_root, w_conv_left, w_conv_right,
                 w_hid, w_out,
                 b_conv, b_hid, b_out,
                 embeddings):
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


def rand_weight(shape_0, shape_1, name):
    with tf.name_scope(name):
        var = tf.Variable(
            tf.truncated_normal(shape=[shape_1, shape_0], stddev=RANDOM_RANGE),
            name=name)
        variable_summaries(var)
    return var


def rand_bias(shape, name):
    return rand_weight(shape, 1, name)


def init_params(author_amount):
    with open('TFAuthorClassifier/embeddings', 'rb') as f:
        np_embs = P.load(f)
    with tf.name_scope('Embeddings'):
        np_embs = OrderedDict(np_embs)
        zero_emb = np.zeros([NUM_FEATURES], np.float32)
        np_embs['ZERO_EMB'] = zero_emb
        emb_indexes = {name: i for i, name in enumerate(np_embs.keys())}
        embeddings = tf.stack(list(np_embs.values()))

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
                  embeddings), emb_indexes


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean, variance = tf.nn.moments(var, [1, 0])
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(variance)
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)
