from collections import namedtuple

NUM_FEATURES = 100

margin = 1
learn_rate = 0.0020  # 0.0001
beta = .0001
momentum = 0.1
l2_param = 5.e-5
clip_const = 1.e-2

NUM_CONVOLUTION = 800  # 50
NUM_DISCRIMINATIVE = 30  # 50
NUM_OUT_LAYER = 10
NUM_POOLING = 3

NUM_HIDDEN = 80

BATCH_SIZE = 1

RANDOM_RANGE = 0.02

NUM_RETRY = 200
NUM_EPOCH = 5000

Network = namedtuple('Network', ['forward', 'back', 'validation'])


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
