import theano.tensor.nnet

from AST.Tokenizer import ast_to_list
from TBCNN.Connection import Connection, PoolConnection
from TBCNN.Layer import Layer, PoolLayer
from TBCNN.NetworkParams import *
import numpy as np
import theano.tensor as T

def compute_leaf_num(root, nodes, depth=0):
    if len(root.children) == 0:
        root.leaf_num = 1
        root.children_num = 1
        return 1, 1, depth  # leaf_num, children_num
    root.allLeafNum = 0
    avg_depth = 0.0
    for child in root.children:
        leaf_num, children_num, child_avg_depth = compute_leaf_num(nodes[child], nodes, depth + 1)
        root.leaf_num += leaf_num
        root.children_num += children_num
        avg_depth += child_avg_depth * leaf_num
    avg_depth /= root.leaf_num
    root.children_num += 1
    return root.leaf_num, root.children_num, avg_depth


def construct_from_ast(ast, parameters: Params):
    nodes = ast_to_list(ast)

    for i in range(len(nodes)):
        if nodes[i].parent is not None:
            nodes[nodes[i].parent].children.append(i)
    for i in range(len(nodes)):
        node = nodes[i]
        len_children = len(node.children)
        if len_children >= 0:
            for child in node.children:
                if len_children == 1:
                    nodes[child].left_rate = .5
                    nodes[child].right_rate = .5
                else:
                    nodes[child].right_rate = nodes[child].pos / (len_children - 1.0)
                    nodes[child].left_rate = 1.0 - nodes[child].right_rate

    dummy, dummy, avg_depth = compute_leaf_num(nodes[-1], nodes)
    avg_depth *= .6
    if avg_depth < 1:        avg_depth = 1

    return construct_network(nodes, parameters, avg_depth)


def construct_network(nodes, parameters: Params, pool_cutoff):
    nodes_amount = len(nodes)
    layers = [Layer] * nodes_amount
    leaf_amount = 0
    for i, node in enumerate(nodes):
        if len(node.children) == 0:
            leaf_amount += 1
        layers[i] = Layer(parameters.embeddings[node.token_index], "embedding_" + str(i))
    not_leaf_amount = nodes_amount - leaf_amount
    layers.extend([Layer] * (2 * not_leaf_amount))
    for i in range(leaf_amount, nodes_amount):
        layers[i + not_leaf_amount] = layers[i]
        layers[i] = Layer(parameters.b_construct, "autoencoder_" + str(i - leaf_amount))

    for i in range(nodes_amount):
        node = nodes[i]
        if node.parent is None: continue
        from_layer = layers[i]
        to_layer = layers[node.parent]
        if node.left_rate != 0:
            Connection(from_layer, to_layer, parameters.w_left,
                       w_coeff=node.left_rate * node.leaf_num / nodes[node.parent].leaf_num)
        if node.right_rate != 0:
            Connection(from_layer, to_layer, parameters.w_right,
                       w_coeff=node.right_rate * node.leaf_num / nodes[node.parent].leaf_num)

    for i in range(leaf_amount, nodes_amount):
        ae_layer = layers[i]
        emb_layer = layers[i + not_leaf_amount]
        layers[i + not_leaf_amount * 2] = ae_layer
        layers[i] = cmb_layer = Layer(None, "combination_" + str(i - leaf_amount))
        Connection(ae_layer, cmb_layer, parameters.w_comb_ae)
        Connection(emb_layer, cmb_layer, parameters.w_comb_emb)

    if (DONT_MAKE_CONV): return layers

    pool_top = PoolLayer('pool_top', NUM_CONVOLUTION)
    pool_left = PoolLayer('pool_left', NUM_CONVOLUTION)
    pool_right = PoolLayer('pool_right', NUM_CONVOLUTION)

    queue = [(nodes_amount - 1, None)]
    layer_cnt = 0

    cur_len = len(queue)
    while cur_len != 0:
        next_queue = []
        for (i, info) in queue:

            cur_layer = layers[i]
            cur_node = nodes[i]

            conv_layer = Layer(parameters.b_conv, "convolve_" + str(i), NUM_CONVOLUTION)
            layers.append(conv_layer)

            Connection(cur_layer, conv_layer, parameters.w_conv_root)

            child_num = len(cur_node.children)

            if layer_cnt < pool_cutoff:
                PoolConnection(conv_layer, pool_top)
            else:
                if info == 'l' or info == 'lr':
                    PoolConnection(conv_layer, pool_left)
                if info == 'r' or info == 'lr':
                    PoolConnection(conv_layer, pool_right)

            for child in cur_node.children:
                child_node = nodes[child]
                child_layer = layers[child]

                if layer_cnt != 0 and info != 'u':
                    child_info = info
                else:
                    root_child_num = len(cur_node.children) - 1
                    if root_child_num == 0:
                        child_info = 'u'
                    elif child_node.pos <= root_child_num / 2.0:
                        child_info = 'l'
                    else:
                        child_info = 'r'

                next_queue.append((child, child_info))

                if child_num == 1:
                    left_w = .5
                    right_w = .5
                else:
                    right_w = child_node.pos / (child_num - 1.0)
                    left_w = 1 - right_w
                if left_w != 0:
                    Connection(child_layer, conv_layer, parameters.w_conv_left, left_w)
                if right_w != 0:
                    Connection(child_layer, conv_layer, parameters.w_conv_right, right_w)

            queue = next_queue

        layer_cnt += 1
        cur_len = len(queue)

    for i in range(leaf_amount, leaf_amount + not_leaf_amount):
        pos = i + 2 * not_leaf_amount
        tmp = layers[i]
        layers[i] = layers[pos]
        layers[pos] = tmp

    dis_layer = Layer(parameters.b_dis, 'discriminative', NUM_DISCRIMINATIVE)

    def softmax(z):
        z -= T.max(z, axis=0)  # extract the maximal value for each dataset to prevent numerical overflow
        z = np.e ** z
        sumbycol = T.sum(z, axis=0)
        return z / sumbycol

    out_layer = Layer(parameters.b_out, "softmax", NUM_OUT_LAYER, softmax)

    Connection(pool_top, dis_layer, parameters.w_dis_top)
    Connection(pool_left, dis_layer, parameters.w_dis_left)
    Connection(pool_right, dis_layer, parameters.w_dis_right)

    Connection(dis_layer, out_layer, parameters.w_out)

    layers.append(pool_top)
    layers.append(pool_left)
    layers.append(pool_right)

    layers.append(dis_layer)
    layers.append(out_layer)

    return layers
