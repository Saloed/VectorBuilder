from theano.compile import SharedVariable as SV

NUM_FEATURES = 30
SAMPLES_AMOUNT = 1000
MARGIN = 1
LEARN_RATE = 0.2
MOMENTUM = 0.1
RANDOM_RANGE = 0.2
EPOCH_IN_RETRY = 1500


class Parameters:
    def __init__(self,
                 w_left: SV, w_right: SV, b_construct: SV, embeddings: list):
        self.w = {
            'w_left': w_left,
            'w_right': w_right
        }
        self.b_construct = b_construct
        self.embeddings = embeddings
