from enum import Enum

from AST.Token import Token
from AST.Tokenizer import Nodes
from NN.TFLayer import *
from NN.TFConection import *
from TFAuthorClassifier.TFParameters import *
import tensorflow as tf


class BuildMode(Enum):
    train = 'train'
    validation = 'validation'
    test = 'test'


class NetLoss:
    def __init__(self, net_forward, loss, error):
        self.net_forward = net_forward
        self.loss = loss
        self.error = error


def compute_rates(root_node: Token):
    if not root_node.is_leaf:
        len_children = len(root_node.children)
        for child in root_node.children:
            if len_children == 1:
                child.left_rate = .5
                child.right_rate = .5
            else:
                child.right_rate = child.pos / (len_children - 1.0)
                child.left_rate = 1.0 - child.right_rate
            compute_rates(child)


# @timing
def construct_from_nodes(ast: Nodes, parameters: Params, mode: BuildMode, target, authors_amount):
    # visualize(ast.root_node, 'ast.png')
    nodes = ast.all_nodes
    compute_rates(ast.root_node)
    compute_leaf_num(ast.root_node, nodes)
    return construct_network(ast, parameters, mode, target, authors_amount)


def compute_leaf_num(root, nodes, depth=0):
    if root.is_leaf:
        root.leaf_num = 1
        root.children_num = 1
        return 1, 1, depth  # leaf_num, children_num, depth
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


# one way pooling (applied now)
def convolve_creator(root_node, pooling_layer, conv_layers, layers, params):
    child_len = len(root_node.children)
    # if child_len == 0: return
    conv_layer = Convolution(params.b['b_conv'], "convolve_" + str(root_node))
    conv_layers.append(conv_layer)
    root_layer = layers[root_node.index]
    Connection(root_layer, conv_layer, params.w['w_conv_root'])
    PoolConnection(conv_layer, pooling_layer)

    for child in root_node.children:
        if child_len == 1:
            left_w = .5
            right_w = .5
        else:
            right_w = child.pos / (child_len - 1.0)
            left_w = 1.0 - right_w

        child_layer = layers[child.index]
        if left_w != 0:
            Connection(child_layer, conv_layer, params.w['w_conv_left'], left_w)
        if right_w != 0:
            Connection(child_layer, conv_layer, params.w['w_conv_right'], right_w)

        convolve_creator(child, pooling_layer, conv_layers, layers, params)


def build_net(nodes: Nodes, params: Params, authors_amount):
    used_embeddings = {}
    nodes_amount = len(nodes.all_nodes)

    _layers = [Layer] * nodes_amount

    for node in nodes.all_nodes:
        emb = params.embeddings[node.token_type]
        used_embeddings[node.token_type] = emb
        _layers[node.index] = Embedding(emb, "embedding_" + str(node))

    conv_layers = []
    pooling_layer = Pooling('pool', NUM_CONVOLUTION)
    convolve_creator(nodes.root_node, pooling_layer, conv_layers, _layers, params)
    hid_layer = FullConnected(params.b['b_hid'], tf.nn.sigmoid, name='hidden_layer', feature_amount=NUM_HIDDEN)
    out_layer = FullConnected(params.b['b_out'], tf.nn.softmax, name='out_layer', feature_amount=authors_amount)
    Connection(pooling_layer, hid_layer, params.w['w_hid'])
    Connection(hid_layer, out_layer, params.w['w_out'])
    layers = _layers + conv_layers
    layers.append(pooling_layer)
    layers.append(hid_layer)
    layers.append(out_layer)
    return layers, used_embeddings, pooling_layer


def construct_network(nodes: Nodes, parameters: Params, mode: BuildMode, target, authors_amount):
    net, used_embeddings, dis_layer = build_net(nodes, parameters, authors_amount)

    def f_builder(layer: Layer):
        if layer.forward is None:
            for c in layer.in_connection:
                if c.forward is None:
                    f_builder(c.from_layer)
                    c.build_forward()
            layer.build_forward()

    def back_propagation(net_forward):
        error = tf.not_equal(tf.argmax(net_forward), target[1])
        loss = -tf.reduce_sum(target[0] * tf.log(net_forward + 1.e-10))
        return NetLoss(net_forward, loss, error)

    f_builder(net[-1])

    net_forward = net[-1].forward
    # pydotprint(net_forward,'net_fwd.jpg',format='jpg')
    # raise Exception('dont need more')

    if mode == BuildMode.train or mode == BuildMode.validation:
        return back_propagation(net_forward)
    else:
        return function([], net_forward)
