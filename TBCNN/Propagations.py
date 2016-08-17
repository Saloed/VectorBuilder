from theano import scan


def forward_propagation(network: list):
    last_layer = network[0]
    for layer in network:
        last_layer = layer
        for con in layer.forward_connection:
            con.f_prop()

    return last_layer.forward()
