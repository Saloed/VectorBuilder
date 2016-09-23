import theano.tensor as T
from theano import function

from AST.Tokenizer import ast_to_nodes, Nodes
from TBCNN.Connection import Connection, PoolConnection
from TBCNN.Layer import *
from TBCNN.NetworkParams import *
from enum import Enum


class NodeInfo(Enum):
    left = 'l'
    right = 'r'
    unknown = 'u'


def construct_from_nodes(ast: Nodes, parameters: Params, need_back_prop=False):
    nodes = ast.all_nodes
    for node in nodes:
        len_children = len(node.children)
        if len_children >= 0:
            for child in node.children:
                if len_children == 1:
                    child.left_rate = .5
                    child.right_rate = .5
                else:
                    child.right_rate = child.pos / (len_children - 1.0)
                    child.left_rate = 1.0 - child.right_rate

    _, _, avg_depth = compute_leaf_num(ast.root_node, nodes)
    avg_depth *= .6
    if avg_depth < 1:        avg_depth = 1

    return construct_network(ast, parameters, need_back_prop, avg_depth)


def compute_leaf_num(root, nodes, depth=0):
    if len(root.children) == 0:
        root.leaf_num = 1
        root.children_num = 1
        return 1, 1, depth  # leaf_num, children_num, depth
    root.allLeafNum = 0
    avg_depth = 0.0
    for child in root.children:
        leaf_num, children_num, child_avg_depth = compute_leaf_num(child, nodes, depth + 1)
        root.leaf_num += leaf_num
        root.children_num += children_num
        avg_depth += child_avg_depth * leaf_num
    avg_depth /= root.leaf_num
    root.children_num += 1
    return root.leaf_num, root.children_num, avg_depth


ConvolveParams = namedtuple('ConvolveParams',
                            ['pool_cutoff', 'layers', 'params', 'pool_top', 'pool_left', 'pool_right'])


def convolve_creator(root_node, node_info: NodeInfo, conv_layers, depth, cp: ConvolveParams):
    child_len = len(root_node.children)
    if child_len == 0: return
    conv_layer = Convolution(cp.params.b['b_conv'], "convolve_" + str(root_node))
    conv_layers.append(conv_layer)
    root_layer = cp.layers[root_node.index]
    Connection(root_layer, conv_layer, cp.params.w['w_conv_root'])
    if depth < cp.pool_cutoff:
        PoolConnection(conv_layer, cp.pool_top)
    else:
        if node_info == NodeInfo.left:
            PoolConnection(conv_layer, cp.pool_left)
        if node_info == NodeInfo.right:
            PoolConnection(conv_layer, cp.pool_right)

    for child in root_node.children:
        if child_len == 1:
            left_w = .5
            right_w = .5
        else:
            right_w = child.pos / (child_len - 1.0)
            left_w = 1.0 - right_w

        child_layer = cp.layers[child.index]
        if left_w != 0:
            Connection(child_layer, conv_layer, cp.params.w['w_conv_left'], left_w)
        if right_w != 0:
            Connection(child_layer, conv_layer, cp.params.w['w_conv_right'], right_w)

        if depth != 0 and node_info != NodeInfo.unknown:
            child_info = node_info
        else:
            child_num = child_len - 1
            if child_num == 0:
                child_info = 'u'
            elif child.pos <= child_num / 2.0:
                child_info = 'l'
            else:
                child_info = 'r'
        convolve_creator(child, child_info, conv_layers, depth + 1, cp)


def build_net(nodes: Nodes, params: Params, pool_cutoff):
    used_embeddings = {}
    nodes_amount = len(nodes.all_nodes)

    _layers = [Layer] * nodes_amount

    for node in nodes.all_nodes:
        emb = params.embeddings[node.token_index]
        used_embeddings[node.token_index] = emb
        _layers[node.index] = Embedding(emb, "embedding_" + str(node))

    for node in nodes.non_leafs:
        emb_layer = _layers[node.index]
        ae_layer = Encoder(params.b['b_construct'], "autoencoder_" + str(node))
        cmb_layer = Combination("combination_" + str(node))
        Connection(ae_layer, cmb_layer, params.w['w_comb_ae'])
        Connection(emb_layer, cmb_layer, params.w['w_comb_emb'])
        _layers[node.index] = cmb_layer
        for child in node.children:
            if child.left_rate != 0:
                Connection(_layers[child.index], ae_layer, params.w['w_left'],
                           w_coeff=child.left_rate * child.leaf_num / node.leaf_num)
            if child.right_rate != 0:
                Connection(_layers[child.index], ae_layer, params.w['w_right'],
                           w_coeff=child.right_rate * child.leaf_num / node.leaf_num)

    pool_top = Pooling('pool_top', NUM_CONVOLUTION)
    pool_left = Pooling('pool_left', NUM_CONVOLUTION)
    pool_right = Pooling('pool_right', NUM_CONVOLUTION)

    conv_params = ConvolveParams(pool_cutoff, _layers, params, pool_top, pool_left, pool_right)
    conv_layers = []

    convolve_creator(nodes.root_node, NodeInfo.unknown, conv_layers, 0, conv_params)

    dis_layer = FullConnected(params.b['b_dis'], activation=T.tanh,
                              name='discriminative', feature_amount=NUM_DISCRIMINATIVE)

    def softmax(z):
        e_z = T.exp(z - z.max(axis=0, keepdims=True))
        return e_z / e_z.sum(axis=0, keepdims=True)

    out_layer = FullConnected(params.b['b_out'], activation=softmax,
                              name="softmax", feature_amount=NUM_OUT_LAYER)

    Connection(pool_top, dis_layer, params.w['w_dis_top'])
    Connection(pool_left, dis_layer, params.w['w_dis_left'])
    Connection(pool_right, dis_layer, params.w['w_dis_right'])

    Connection(dis_layer, out_layer, params.w['w_out'])

    layers = _layers + conv_layers

    layers.append(pool_top)
    layers.append(pool_left)
    layers.append(pool_right)

    layers.append(dis_layer)
    layers.append(out_layer)
    return layers


def construct_network(nodes: Nodes, parameters: Params, need_back_prop: bool, pool_cutoff):
    net = build_net(nodes, parameters, pool_cutoff)

    def f_builder(layer: Layer):
        if layer.forward is None:
            for c in layer.in_connection:
                if c.forward is None:
                    f_builder(c.from_layer)
                    c.build_forward()
            layer.build_forward()

    def back_propagation(net_forward):
        if not need_back_prop:
            return None, None
        pass

    f_builder(net[-1])
    net_forward = net[-1].forward
    net_back, net_validation = back_propagation(net_forward)
    return Network(net_forward, net_back, net_validation)
