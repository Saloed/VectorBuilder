from theano import function
import theano.tensor as T
from Embeddings.Parameters import Parameters, NUM_FEATURES, LEARN_RATE
from AST.TokenMap import token_map
import numpy as np

from TBCNN.Connection import Connection
from TBCNN.Layer import Layer
from TBCNN.NetworkParams import Network, Updates
from TBCNN.Builder import compute_leaf_num


def construct(tokens, params: Parameters, update: Updates):
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

    compute_leaf_num(tokens[-1], tokens)

    nodes_amount = len(tokens)
    layers = [Layer] * nodes_amount
    leaf_amount = 0
    for i, node in enumerate(tokens):
        if len(node.children) == 0:
            leaf_amount += 1
        layers[i] = Layer(params.embeddings[node.token_index], "embedding_" + str(i))

    for i in range(nodes_amount):
        node = tokens[i]
        if node.parent is None: continue
        if node.parent >= nodes_amount: continue

        from_layer = layers[i]
        to_layer = layers[node.parent]

        if node.left_rate != 0:
            Connection(from_layer, to_layer, params.w['w_left'],
                       w_coeff=node.left_rate * node.leaf_num / tokens[node.parent].leaf_num)
        if node.right_rate != 0:
            Connection(from_layer, to_layer, params.w['w_right'],
                       w_coeff=node.right_rate * node.leaf_num / tokens[node.parent].leaf_num)

    def f_builder(layer):
        if not layer.f_initialized:
            for c in layer.in_connection:
                if not c.f_initialized:
                    f_builder(c.from_layer)
                    c.build_forward()
            layer.build_forward()

    def b_builder(layer, update: Updates):
        if not layer.b_initialized:
            for c in layer.out_connection:
                if not c.b_initialized:
                    b_builder(c.to_layer, update)
                    c.build_back(update)
            layer.build_back(update)

    def forward_propagation(network: list):
        last_layer = network[-1]
        return function([], last_layer.forward)

    def back_propagation(updates: Updates):
        update_w = []
        update_b = []

        def make_update(target, upd):
            print(target)
            print(upd.type)
            tpl = (target, target + T.mul(upd, LEARN_RATE))
            return tpl

        for (bias, upd) in updates.bias_updates.items():
            update_b.append(make_update(updates.grad_b[bias], upd))

        for (weights, upd) in updates.weights_updates.items():
            update_w.append(make_update(updates.grad_w[weights], upd))
        update = update_w
        update.extend(update_b)
        return function([updates.error], updates=update, on_unused_input='ignore')

    network = Network(layers)

    for lay in layers:
        f_builder(lay)
    network.forward = forward_propagation(layers)

    for lay in reversed(layers):
        b_builder(lay, update)
    network.back = back_propagation(update)

    return network
