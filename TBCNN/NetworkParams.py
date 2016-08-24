NUM_FEATURES = 30

margin = 1
learn_rate = 0.0025  # 0.0001
beta = .0001
momentum = 0.1

NUM_CONVOLUTION = 600  # 50
NUM_DISCRIMINATIVE = 600  # 50
NUM_OUT_LAYER = 104
NUM_POOLING = 3

# test parameter
DONT_MAKE_CONV = False

BATCH_SIZE = 1


class Updates:
    def __init__(self):
        self.bias_updates = dict()
        self.weights_updates = dict()


class Params:
    def __init__(self, weights, biases,
                 w_left, w_right,
                 w_comb_ae, w_comb_emb,
                 w_conv_root, w_conv_left, w_conv_right,
                 w_dis_top, w_dis_left, w_dis_right,
                 w_out,
                 b_token, b_construct,
                 b_conv, b_dis, b_out,
                 embeddings):
        self.weights = weights
        self.biases = biases

        self.w_left = w_left
        self.w_right = w_right

        self.w_comb_ae = w_comb_ae
        self.w_comb_emb = w_comb_emb

        self.w_conv_root = w_conv_root
        self.w_conv_left = w_conv_left
        self.w_conv_right = w_conv_right

        self.w_dis_top = w_dis_top
        self.w_dis_left = w_dis_left
        self.w_dis_right = w_dis_right

        self.w_out = w_out

        self.b_token = b_token
        self.b_construct = b_construct

        self.b_conv = b_conv
        self.b_dis = b_dis
        self.b_out = b_out

        self.embeddings = embeddings
