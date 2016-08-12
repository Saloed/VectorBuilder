class Connection:
    x_layer = None
    y_layer = None
    x_num = None
    y_num = None
    weights = None  # reshape to lnum by unum
    w_coef = None
    y_down_id = 0

    def __init__(self, x_layer, y_layer, x_num, y_num, weights, w_coef=1.0):
        self.x_layer = x_layer
        self.y_layer = y_layer
        self.x_num = x_num
        self.y_num = y_num
        self.weights = weights
        self.w_coef = w_coef
        self.x_layer.connectUp.append(self)
        self.y_layer.connectDown.append(self)