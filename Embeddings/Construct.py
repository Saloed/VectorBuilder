from copy import deepcopy
import numpy as np
import theano.tensor as T
from theano import function

from AST.TokenMap import token_map
from Embeddings.Parameters import Parameters, MARGIN
from TBCNN.Builder import compute_leaf_num
from TBCNN.Connection import Connection
from TBCNN.Layer import Layer
from theano.compile import SharedVariable as TS

from Utils.Wrappers import timing


def compute_rates(tokens):
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


def random_change(tokens, layers, params, root_token, used_embeddings):
    def rand_token():
        return list(token_map.keys())[np.random.randint(0, len(token_map))]

    swap_token_index = np.random.randint(0, len(tokens))
    current = tokens[swap_token_index]
    new_token = rand_token()
    while current.token_type == new_token:
        new_token = rand_token()
    new_token_index = token_map[new_token]
    new_emb = params.embeddings[new_token_index]

    if swap_token_index == root_token:
        return new_emb
    else:
        if new_emb not in used_embeddings:
            used_embeddings[new_token_index] = new_emb
        layers[swap_token_index].bias = new_emb
        return None


def construct(tokens, params: Parameters, root_token_index, just_validation=False):
    compute_rates(tokens)
    compute_leaf_num(tokens[root_token_index], tokens)
    nodes_amount = len(tokens)
    # layer index equal token index
    layers = [Layer] * nodes_amount
    used_embeddings = dict()

    assert root_token_index == 0

    root_token = tokens[root_token_index]
    positive_target = params.embeddings[root_token.token_index]

    root_layer = Layer(params.b_construct, "root_layer")
    layers[root_token_index] = root_layer
    used_embeddings[root_token.token_index] = positive_target

    for i, node in enumerate(tokens):
        if i == root_token_index:
            continue
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

    neg_layers = deepcopy(layers)

    negative_target = random_change(tokens, neg_layers, params, root_token_index,
                                    used_embeddings)

    del used_embeddings[root_token.token_index]

    def f_builder(layer: Layer):
        if layer.forward is None:
            for c in layer.in_connection:
                if c.forward is None:
                    f_builder(c.from_layer)
                    c.build_forward()
            layer.build_forward()

    # def forward_propagation(network_layers: list):
    #     f_builder(network_layers[root_token_index])
    #     return function([], network_layers[root_token_index].forward)

    # @timing
    def back_propagation(pos_forward, neg_forward):
        alpha = T.fscalar('alpha')
        decay = T.fscalar('decay')
        pos_target = positive_target
        neg_target = negative_target

        upd_params = [params.b_construct]
        upd_params.extend(params.w.values())
        upd_params.extend(used_embeddings.values())

        pos_delta = pos_target - pos_forward

        if neg_target is None:
            neg_delta = neg_forward - pos_target
        else:
            neg_delta = neg_forward - neg_target

        pos_d = T.mul(T.sum(T.mul(pos_delta, pos_delta)), 0.5)
        neg_d = T.mul(T.sum(T.mul(neg_delta, neg_delta)), 0.5)
        error = T.nnet.relu(MARGIN + pos_d - neg_d)

        if not just_validation:
            gparams = [T.grad(error, param) for param in upd_params]
            updates = [
                (param, param - alpha * gparam - decay * alpha * param)
                for param, gparam in zip(upd_params, gparams)
                ]

            updates.append((pos_target, pos_target - pos_delta))
            if neg_target is not None:
                updates.append((neg_target, neg_target - neg_delta))

            return function([alpha, decay], outputs=error, updates=updates)
        else:
            return function([alpha, decay], outputs=error, on_unused_input='ignore')

    f_builder(layers[root_token_index])
    f_builder(neg_layers[root_token_index])

    pos_forward = layers[root_token_index].forward
    neg_forward = neg_layers[root_token_index].forward

    back_prop = back_propagation(pos_forward, neg_forward)

    return back_prop
