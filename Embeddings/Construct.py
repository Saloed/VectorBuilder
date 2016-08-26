from theano import function

from Embeddings.Parameters import Parameters, NUM_FEATURE
from AST.TokenMap import token_map
import numpy as np

from TBCNN.Connection import Connection
from TBCNN.Layer import Layer
from TBCNN.NetworkParams import Network, Updates


def construct(tokens, params: Parameters):
    # todo add fill of array
    leaf_cnt = np.array([])

    layers = []

    root_token = tokens[0]
    root_idx = token_map[root_token]
    root = Layer(bias=params.embeddings[root_idx], name=root_token, feature_amount=NUM_FEATURE)

    child_amount = len(tokens)
    for i in range(1, child_amount):
        child_token = tokens[i]
        child_idx = token_map[child_token]

        child = Layer(bias=params.embeddings[child_idx], name=child_token, feature_amount=NUM_FEATURE)

        if child_amount == 2:
            left_coeff = 0.5
            right_coeff = 0.5
        else:
            right_coeff = (i - 1) / (child_amount - 2)
            left_coeff = 1 - right_coeff

        left_coeff *= leaf_cnt[i - 1]
        right_coeff *= leaf_cnt[i - 1]

        if left_coeff != 0:
            Connection(child, root, params.w_left, left_coeff)
        if right_coeff != 0:
            Connection(child, root, params.w_right, right_coeff)

        layers.append(child)
    layers.append(root)

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
        forward = function([], last_layer.forward)
        return forward

    def back_propagation(updates: Updates):
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
        return b_prop

    network = Network(layers)

    for lay in layers:
        f_builder(lay)
    network.forward = forward_propagation(layers)

    update = Updates()
    for lay in reversed(layers):
        b_builder(lay, update)
    network.back = back_propagation(update)

    return network
