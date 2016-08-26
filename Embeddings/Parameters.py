from theano.compile import SharedVariable as SV

NUM_FEATURE = 30


class Parameters:
    def __init__(self,
                 w_left: SV, w_right: SV, embeddings):
        self.w_left = w_left
        self.w_right = w_right
        self.embeddings = embeddings
