from theano.compile import SharedVariable as SV

NUM_FEATURES = 30
MARGIN = 1
LEARN_RATE = 0.003
MOMENTUM = 0.1
RANDOM_RANGE = 0.2

class Parameters:
    def __init__(self,
                 w_left: SV, w_right: SV, b_construct: SV, embeddings: list):
        self.w = {
            'w_left': w_left,
            'w_right': w_right
        }
        self.b_construct = b_construct
        self.embeddings = embeddings
