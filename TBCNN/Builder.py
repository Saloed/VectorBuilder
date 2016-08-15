from AST.Tokenizer import ast_to_list
from TBCNN.Connection import Connection
from TBCNN.Layer import Layer
from TBCNN.NetworkParams import *


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

    return layers