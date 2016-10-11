from copy import deepcopy

from lasagne.updates import adadelta
from theano import function

from AuthorClassifier.Builder import compute_leaf_num
from Embeddings.Parameters import Parameters, MARGIN
from NN.Connection import Connection
from NN.Layer import *


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
                    child.right_rate = (child.pos - 1.0) / (len_children - 1.0)
                    child.left_rate = 1.0 - child.right_rate


def build_net(tokens, params: Parameters, root_token_index, used_embeddings, change_index=-1, secret_param=None):
    nodes_amount = len(tokens)
    # layer index equal token index
    layers = [Layer] * nodes_amount
    root_token = tokens[root_token_index]

    assert root_token_index == 0
    assert root_token.index == root_token_index

    positive_target = params.embeddings[root_token.token_type]
    used_embeddings[root_token.token_type] = positive_target
    root_layer = Encoder(params.b_construct, "root_layer")
    layers[root_token.index] = root_layer
    used_embeddings['b_construct'] = params.b_construct

    for node in tokens:
        if node.index == root_token_index:
            continue
        if node.index == change_index:
            layers[node.index] = Embedding(secret_param, "embedding_" + secret_param.name)
        else:
            emb = params.embeddings[node.token_type]
            used_embeddings[node.token_type] = emb
            layers[node.index] = Embedding(emb, "embedding_" + str(node))

    for node in tokens:
        if node.parent is None: continue

        from_layer = layers[node.index]
        to_layer = layers[node.parent.index]

        if node.left_rate != 0:
            Connection(from_layer, to_layer, params.w['w_left'],
                       w_coeff=node.left_rate * node.leaf_num / node.parent.leaf_num)
        if node.right_rate != 0:
            Connection(from_layer, to_layer, params.w['w_right'],
                       w_coeff=node.right_rate * node.leaf_num / node.parent.leaf_num)

    return positive_target, layers


def construct(tokens, params: Parameters, root_token_index, just_validation=False):
    work_tokens = deepcopy(tokens)
    compute_rates(work_tokens)
    compute_leaf_num(work_tokens[root_token_index], work_tokens)
    used_embeddings = {}
    positive_target, positive_layers = build_net(work_tokens, params, root_token_index, used_embeddings)
    change_index = np.random.randint(root_token_index + 1, len(tokens))

    secret_param = T.fvector('secret')
    negative_target, negative_layers = build_net(work_tokens, params, root_token_index, used_embeddings, change_index,
                                                 secret_param)

    def f_builder(layer: Layer):
        if layer.forward is None:
            for c in layer.in_connection:
                if c.forward is None:
                    f_builder(c.from_layer)
                    c.build_forward()
            layer.build_forward()

    # @timing
    def back_propagation(pos_forward, neg_forward):
        alpha = T.fscalar('alpha')
        decay = T.fscalar('decay')

        pos_target = positive_target
        neg_target = negative_target

        pos_delta = pos_forward - pos_target
        neg_delta = neg_forward - neg_target

        pos_d = T.std(pos_delta) * 0.5
        neg_d = T.std(neg_delta) * 0.5

        # p_len = T.std(pos_forward)
        # n_len = T.std(neg_forward)

        error = T.nnet.relu(MARGIN + pos_d - neg_d)

        if not just_validation:
            update_params = list(params.w.values()) + list(used_embeddings.values())

            updates = adadelta(error, update_params)

            return function([secret_param, alpha, decay], outputs=error, updates=updates, on_unused_input='ignore')
        else:
            return function([secret_param], outputs=error, on_unused_input='ignore')

    f_builder(positive_layers[root_token_index])
    f_builder(negative_layers[root_token_index])

    pos_forward = positive_layers[root_token_index].forward
    neg_forward = negative_layers[root_token_index].forward

    back_prop = back_propagation(pos_forward, neg_forward)

    return back_prop
