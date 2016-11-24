import theano.tensor as T
import theano.tensor.extra_ops
from lasagne.updates import adadelta, nesterov_momentum, sgd, adam
from theano import function
from lasagne.objectives import *
from theano.printing import pydotprint
from AST.Token import Token
from AST.Tokenizer import Nodes, print_ast, visualize
from NN.Connection import Connection, PoolConnection
from NN.Layer import *
from AuthorClassifier.ClassifierParams import *
from theano.compile.nanguardmode import NanGuardMode
from enum import Enum

from Utils.Wrappers import timing


class BuildMode(Enum):
    train = 'train'
    validation = 'validation'
    test = 'test'


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
def construct_from_nodes(ast: Nodes, parameters: Params, mode: BuildMode, author_amount):
    # visualize(ast.root_node, 'ast.png')
    nodes = ast.all_nodes
    compute_rates(ast.root_node)
    _, _, avg_depth = compute_leaf_num(ast.root_node, nodes)
    avg_depth *= .6
    if avg_depth < 1:        avg_depth = 1

    return construct_network(ast, parameters, mode, avg_depth, author_amount)


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


def build_net(nodes: Nodes, params: Params, pool_cutoff, authors_amount):
    used_embeddings = {}
    nodes_amount = len(nodes.all_nodes)

    _layers = [Layer] * nodes_amount

    for node in nodes.all_nodes:
        emb = params.embeddings[node.token_type]
        used_embeddings[node.token_type] = emb
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

    conv_layers = []
    pooling_layer = Pooling('pool', NUM_CONVOLUTION)
    convolve_creator(nodes.root_node, pooling_layer, conv_layers, _layers, params)
    out_layer = FullConnected(params.svm['b_out'], T.nnet.sigmoid, name='out_layer', feature_amount=1)
    Connection(pooling_layer, out_layer, params.svm['w_out'])
    layers = _layers + conv_layers
    layers.append(pooling_layer)
    layers.append(out_layer)
    return layers, used_embeddings, pooling_layer


def construct_network(nodes: Nodes, parameters: Params, mode: BuildMode, pool_cutoff, author_amount):
    parameters_amount = {}
    for k in parameters.w.keys():
        parameters_amount[k] = 0
    for k in parameters.b.keys():
        parameters_amount[k] = 0

    net, used_embeddings, dis_layer = build_net(nodes, parameters, pool_cutoff, author_amount)

    def f_builder(layer: Layer):
        if layer.forward is None:
            for c in layer.in_connection:
                if c.forward is None:
                    f_builder(c.from_layer)
                    c.build_forward()
            layer.build_forward()

    def back_propagation(net_forward):
        target = T.iscalar('target')
        error = T.neq(T.round(net_forward), target)
        # cost = T.sqrt(T.mean(T.sqr(net_forward - target) + 1e-10))
        cost = -T.max(target * T.log(net_forward + 1e-10) + (1 - target) * T.log(1 - net_forward + 1e-10))

        if mode == BuildMode.train:
            params_keys = ['w_conv_left', 'w_conv_right', 'w_conv_root', 'w_comb_ae', 'w_comb_emb']
            used_params = [parameters.w[k] for k in params_keys]
            params_keys.append('b_conv')
            used_params.append(parameters.b['b_conv'])

            used_params.extend(parameters.svm.values())

            updates = sgd(cost, used_params, 0.001)

            return function([target], [cost, error, net_forward], updates=updates)
        else:
            return function([target], [cost, error, net_forward])

    f_builder(net[-1])

    net_forward = net[-1].forward
    tbcnn_out = dis_layer.forward

    # pydotprint(net_forward,'net_fwd.jpg',format='jpg')
    # raise Exception('dont need more')

    if mode == BuildMode.train or mode == BuildMode.validation:
        return back_propagation(net_forward)
    else:
        return function([], net_forward)
