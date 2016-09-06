import theano.tensor as T
from theano import function

from Embeddings.Parameters import Parameters, MARGIN
from TBCNN.Builder import compute_leaf_num
from TBCNN.Connection import Connection
from TBCNN.Layer import Layer
from TBCNN.NetworkParams import Network


def construct(tokens, params: Parameters, root_token_index, is_negative=False):
    for i in range(len(tokens)):
        node = tokens[i]
        len_children = len(node.children)
        if len_children >= 0:
            for child in node.children:
                if len_children == 1:
                    child.left_rate = .5
                    child.right_rate = .5
                else:
                    child.right_rate = child.pos / (len_children - 1.0)
                    child.left_rate = 1.0 - child.right_rate

    compute_leaf_num(tokens[root_token_index], tokens)
    used_embeddings = dict()
    nodes_amount = len(tokens)
    # layer index equal token index
    layers = [Layer] * nodes_amount
    leaf_amount = 0
    for i, node in enumerate(tokens):
        if len(node.children) == 0:
            leaf_amount += 1
        emb = params.embeddings[node.token_index]
        used_embeddings[node.token_index] = emb
        layers[i] = Layer(emb, "embedding_" + str(i))

    for i in range(nodes_amount):
        node = tokens[i]
        if node.parent is None: continue

        from_layer = layers[i]
        to_layer = layers[node.parent]

        if node.left_rate != 0:
            Connection(from_layer, to_layer, params.w['w_left'],
                       w_coeff=node.left_rate * node.leaf_num / tokens[node.parent].leaf_num)
        if node.right_rate != 0:
            Connection(from_layer, to_layer, params.w['w_right'],
                       w_coeff=node.right_rate * node.leaf_num / tokens[node.parent].leaf_num)

    def f_builder(layer: Layer):
        if layer.forward is None:
            for c in layer.in_connection:
                if c.forward is None:
                    f_builder(c.from_layer)
                    c.build_forward()
            layer.build_forward()

    def forward_propagation(network_layers: list):
        f_builder(network_layers[root_token_index])
        return function([], network_layers[root_token_index].forward)

    def back_propagation(network_layers: list):
        alpha = T.fscalar('alpha')
        target = T.fvector('Targ')
        opposite_forward = T.fvector('F')

        parameters = []
        parameters.extend(used_embeddings.values())
        parameters.extend(params.w.values())

        delta = network_layers[root_token_index].forward - target
        op_delta = opposite_forward - target

        mse = T.mul(T.sum(T.mul(delta, delta)), 0.5)
        op_mse = T.mul(T.sum(T.mul(op_delta, op_delta)), 0.5)

        if is_negative:
            error = MARGIN + op_mse - mse
        else:
            error = MARGIN + mse - op_mse

        gparams = [T.grad(error, param) for param in parameters]
        updates = [
            (param, param - alpha * gparam)
            for param, gparam in zip(parameters, gparams)
            ]
        return function([target, opposite_forward, alpha], updates=updates)

    network = Network()
    network.forward = forward_propagation(layers)
    network.back = back_propagation(layers)

    return network
