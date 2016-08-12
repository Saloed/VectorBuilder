import numpy as np
import theano

from AST.Tokenizer import ast_to_list
from TBCNN.Connection import Connection
from TBCNN.Layer import Layer


def construct_network(nodes,
                      param):
    num_nodes = len(nodes)

    # TODO fix this
    ###############################
    num_fea = 30

    emb_weights = np.tile(theano.shared(np.asarray(), 'EW'), num_nodes)
    emb_bias = np.tile(theano.shared(np.asarray(), 'EB'), num_nodes)

    ae_bias = np.tile(theano.shared(np.asarray(), 'AEB'), num_nodes)
    ae_weights = np.tile(theano.shared(np.asarray(), 'AEW'), num_nodes)

    w_left = np.tile(theano.shared(np.asarray(), 'WL'), num_nodes)
    w_right = np.tile(theano.shared(np.asarray(), 'WR'), num_nodes)

    w_comb_ae = np.tile(theano.shared(np.asarray(), 'WCA'), num_nodes)
    w_comb_emb = np.tile(theano.shared(np.asarray(), 'WCE'), num_nodes)

    ##############################

    layers = [Layer] * num_nodes

    # embedding layers
    num_leaf = 0
    for idx in range(num_nodes):
        node = nodes[idx]
        if len(node.children) == 0:
            num_leaf += 1
        layers[idx] = Layer(emb_bias[idx], 'embedding')

    # autoencoding layers

    num_non_leaf = num_nodes - num_leaf

    layers.extend([Layer] * (2 * num_non_leaf))

    for idx in range(num_leaf, num_nodes):
        layers[idx + num_non_leaf] = layers[idx]
        tmp_layer = Layer(ae_bias, 'autoencoding')
        layers[idx] = tmp_layer

    # add reconstruction connections
    for idx in range(num_nodes):
        node = nodes[idx]
        if node.parent is None:
            continue
        tmplayer = layers[idx]
        parent = layers[node.parent]
        if node.leftRate != 0:
            Connection(tmplayer, parent,
                       num_fea, num_fea, w_left,
                       w_coef=node.leftRate * node.leafNum / nodes[node.parent].leafNum)
        if node.rightRate != 0:
            Connection(tmplayer, parent,
                       num_fea, num_fea, w_right,
                       w_coef=node.rightRate * node.leafNum / nodes[node.parent].leafNum)

    for idx in range(num_leaf, num_nodes):
        ae_layer = layers[idx]
        emb_layer = layers[idx + num_non_leaf]
        layers[idx + num_non_leaf * 2] = ae_layer

        comlayer = Layer(None, 'combination')
        layers[idx] = comlayer
        # connecton auto encoded vector and original vector
        Connection(ae_layer, comlayer, num_fea, num_fea, w_comb_ae)
        Connection(emb_layer, comlayer, num_fea, num_fea, w_comb_emb)

    return layers


def construct_from_ast(ast):
    nodes = ast_to_list(ast)
    for idx in range(len(nodes)):
        if nodes[idx].parent is not None:
            nodes[nodes[idx].parent].children.append(idx)

    for idx in range(len(nodes)):
        node = nodes[idx]
        child_amount = len(node.children)

        for child in node.children:
            if child_amount == 1:
                nodes[child].left_rate = .5
                nodes[child].right_rate = .5
            else:
                nodes[child].right_rate = nodes[child].pos / (child_amount - 1.0)
                nodes[child].left_rate = 1.0 - nodes[child].right_rate

    dummy, dummy, avg_depth = compute_leaf_num(nodes[-1], nodes)

    avg_depth *= .6
    if avg_depth < 1: avg_depth = 1

    network = construct_network(
        nodes,
        # TODO initial parameters
        params
    )

    return network


def compute_leaf_num(root, nodes, depth=0):
    if len(root.children) == 0:
        root.leafNum = 1
        root.childrenNum = 1
        return 1, 1, depth  # leaf_num, children_num
    root.allLeafNum = 0
    avg_depth = 0.0
    for child in root.children:
        leaf_num, children_num, child_avg_depth = compute_leaf_num(nodes[child], nodes, depth + 1)
        root.leafNum += leaf_num
        root.childrenNum += children_num
        avg_depth += child_avg_depth * leaf_num
    avg_depth /= root.leafNum
    root.childrenNum += 1
    return root.leafNum, root.childrenNum, avg_depth
