num_features = 30

margin = 1
learn_rate = 0.0025  # 0.0001
beta = .0001
momentum = 0.1

num_convolution = 600  # 50
num_discriminative = 600  # 50
num_out_layer = 104
num_pooling = 3


class Params:
    w_left = None
    w_right = None

    w_comb_ae = None
    w_comb_emb = None

    w_conv_root = None
    w_conv_left = None
    w_conv_right = None

    w_dis = None

    w_out = None

    b_token = None

    b_construct = None

    b_conv = None

    b_dis = None

    b_out = None

    biases = None
    weights = None

    embeddings = None

    def __init__(self, weights, biases,
                 w_left, w_right,
                 w_comb_ae, w_comb_emb,
                 w_conv_root, w_conv_left, w_conv_right,
                 w_dis, w_out,
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

        self.w_dis = w_dis
        self.w_out = w_out

        self.b_token = b_token
        self.b_construct = b_construct

        self.b_conv = b_conv
        self.b_dis = b_dis
        self.b_out = b_out

        self.embeddings = embeddings
