import theano.tensor as T


class Layer:

    #layer parameters
    bias=None
    z=None

    activation_function=T.nnet.relu

    #layer out
    y=None

    #connections
    connection_up=[]
    connection_down=[]

    next_layer=None
    prev_layer=None

    """
    weights - matrix of weights
    bias   - matrix of biases
    name - name of layer
    """
    def __init__(self, bias, name=None):
        self.name = name
        self.bias = bias

    def compute_y(self,bias,num_data):
