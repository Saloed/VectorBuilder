class Connection:
    def __init__(self, from_layer, to_layer,
                 weights, w_coeff=1.0):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.weights = weights
        self.w_coeff = w_coeff
