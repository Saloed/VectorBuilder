from  theano import function


def forward_propagation(network: list):
    # last_layer = network[0]
    #
    # f = open('out.txt', mode='w')
    #
    # for layer in network:
    #     last_layer = layer
    #     layer.forward()
    #
    #     print("\n" + layer.name + "\n", file=f)
    #     print(layer.forward(), file=f)
    #     print("\n###################\n", file=f)
    #
    # f.close()

    last_layer = network[-1]
    forward = function([], last_layer.forward)
    return forward()
