from  theano import function
from TBCNN.NetworkParams import Updates


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


def back_propagation(network: list, updates: Updates):
    update = []
    diff = []

    def make_update(target, upd):
        diff.append(upd)
        tpl = (target, target + upd)
        return tpl

    for (bias, upd) in updates.bias_updates.items():
        update.append(make_update(bias, upd))

    for (weights, upd) in updates.weights_updates.items():
        update.append(make_update(weights, upd))

    b_prop = function([], updates=update)
    b_prop()

    # test_b_prop = function([], outputs=diff)
    # print(test_b_prop())
