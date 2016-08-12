from TBCNN.Layer import Layer


# TODO rework with theano.scan

def forward_propagation(layer0: Layer, X, W, b):
    if X is not None:
        num_data = X.shape[1]
        layer0.y = X
        layer0.z = X
    else:
        num_data = 1
    # with size numUnitCur x numData
    cur_layer = layer0
    while True:
        # apply the activation function
        cur_layer.compute_y(b, num_data)

        if cur_layer.next_layer is None:
            return cur_layer.y
        # feed forward
        for con in cur_layer.connection_up:
            con.forwardprop(W, num_data)
        cur_layer = cur_layer.next_layer


def backpropagation(outlayer: Layer, Weights, Biases, gradWeights, gradBiases):
    num_data = outlayer.y.shape[1]
    cur_layer = outlayer
    while cur_layer is not None:  # for each layer
        # dE/dy has size <numOutput> by <num_data>
        # if cur_layer.dE_by_dz == None and cur_layer.z != None:
        if cur_layer.dE_by_dy is None:
            cur_layer = cur_layer.prev_layer
            continue
        cur_layer.updateB(gradBiases)

        # back propogation
        for con in cur_layer.connection_down:
            con.backprop(Weights, gradWeights, num_data)
        # end of each upward connection
        cur_layer = cur_layer.prev_layer
    # end of all layers

    pass
