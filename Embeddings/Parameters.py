from theano.compile import SharedVariable as SV

NUM_FEATURES = 30
SAMPLES_AMOUNT = 5600
MARGIN = 1
LEARN_RATE = 0.005
MOMENTUM = 0.1
RANDOM_RANGE = 0.2
EPOCH_IN_RETRY = 3000
NUM_RETRY = 1000
BATCH_SIZE = 10


class Parameters:
    def __init__(self,
                 w_left: SV, w_right: SV, b_construct: SV, embeddings: dict):
        self.w = {
            'w_left': w_left,
            'w_right': w_right
        }
        self.b_construct = b_construct
        self.embeddings = embeddings
