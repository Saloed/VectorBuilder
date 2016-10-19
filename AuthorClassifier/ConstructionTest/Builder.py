import theano.tensor as T
import theano.tensor.extra_ops
from lasagne.updates import adadelta, nesterov_momentum
from theano import function
from lasagne.objectives import *
from theano.printing import pydotprint

from AST.Token import Token
from AST.Tokenizer import Nodes, print_ast, visualize
from NN.Connection import Connection, PoolConnection
from NN.Layer import *
from AuthorClassifier.ClassifierParams import *
from enum import Enum


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


# three way pooling (not work)
# def convolve_creator(root_node, node_info, conv_layers, depth, cp: ConvolveParams):
#     child_len = len(root_node.children)
#     if child_len == 0: return
#     conv_layer = Convolution(cp.params.b['b_conv'], "convolve_" + str(root_node))
#     conv_layers.append(conv_layer)
#     root_layer = cp.layers[root_node.index]
#     Connection(root_layer, conv_layer, cp.params.w['w_conv_root'])
#     if depth < cp.pool_cutoff:
#         PoolConnection(conv_layer, cp.pool_top)
#     else:
#         if node_info == 'left':
#             PoolConnection(conv_layer, cp.pool_left)
#         if node_info == 'right':
#             PoolConnection(conv_layer, cp.pool_right)
#
#     for child in root_node.children:
#         if child_len == 1:
#             left_w = .5
#             right_w = .5
#         else:
#             right_w = child.pos / (child_len - 1.0)
#             left_w = 1.0 - right_w
#
#         child_layer = cp.layers[child.index]
#         if left_w != 0:
#             Connection(child_layer, conv_layer, cp.params.w['w_conv_left'], left_w)
#         if right_w != 0:
#             Connection(child_layer, conv_layer, cp.params.w['w_conv_right'], right_w)
#
#         if depth != 0 and node_info != 'unknown':
#             child_info = node_info
#         else:
#             child_num = child_len - 1
#             if child_num == 0:
#                 child_info = 'unknown'
#             elif child.pos <= child_num / 2.0:
#                 child_info = 'left'
#             else:
#                 child_info = 'right'
#         convolve_creator(child, child_info, conv_layers, depth + 1, cp)

# one way pooling (applied now)
def convolve_creator(root_node, pooling_layer, conv_layers, layers, params):
    child_len = len(root_node.children)
    if child_len == 0: return
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

    for child in root_node.children:
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

    # pool_top = Pooling('pool_top', NUM_CONVOLUTION)
    # pool_left = Pooling('pool_left', NUM_CONVOLUTION)
    # pool_right = Pooling('pool_right', NUM_CONVOLUTION)

    # conv_params = ConvolveParams(pool_cutoff, _layers, params, pool_top, pool_left, pool_right)
    conv_layers = []

    # convolve_creator(nodes.root_node, 'unknown', conv_layers, 0, conv_params)

    pooling_layer = Pooling('pool', NUM_CONVOLUTION)
    convolve_creator(nodes.root_node, pooling_layer, conv_layers, _layers, params)

    dis_layer = FullConnected(params.b['b_dis'], activation=T.tanh,
                              name='discriminative', feature_amount=NUM_DISCRIMINATIVE)

    # def softmax(x):
    #     e_x = T.exp(x - x.max(axis=0, keepdims=True))
    #     return e_x / e_x.sum(axis=0, keepdims=True)

    def logSoftmax(x):
        xdev = x - x.max(axis=0, keepdims=True)
        return xdev - T.log(T.sum(T.exp(xdev), axis=0, keepdims=True))

    out_layer = FullConnected(params.b['b_out'],  # activation=lambda x: x,
                              activation=logSoftmax,  # T.nnet.softmax,   # not work (????)
                              name="softmax", feature_amount=authors_amount)

    Connection(pooling_layer, dis_layer, params.w['w_dis_top'])

    # Connection(pool_top, dis_layer, params.w['w_dis_top'])
    # Connection(pool_left, dis_layer, params.w['w_dis_left'])
    # Connection(pool_right, dis_layer, params.w['w_dis_right'])

    Connection(dis_layer, out_layer, params.w['w_out'])

    layers = _layers + conv_layers

    # layers.append(pool_top)
    # layers.append(pool_left)
    # layers.append(pool_right)

    layers.append(pooling_layer)

    layers.append(dis_layer)
    layers.append(out_layer)
    return layers, used_embeddings


# # numerically stable log-softmax with crossentropy
# xdev = x-x.max(1,keepdims=True)
# lsm = xdev - T.log(T.sum(T.exp(xdev),axis=1,keepdims=True))
# sm2 = T.exp(lsm) # just used to show equivalence with sm
# cm2=-T.sum(y*lsm,axis=1)
# g2 = T.grad(cm2.mean(),x)

def construct_network(nodes: Nodes, parameters: Params, mode: BuildMode, pool_cutoff, author_amount):
    net, used_embeddings = build_net(nodes, parameters, pool_cutoff, author_amount)

    # print(pool_cutoff)

    def f_builder(layer: Layer):
        if layer.forward is None:
            for c in layer.in_connection:
                if c.forward is None:
                    f_builder(c.from_layer)
                    c.build_forward()
            layer.build_forward()

    # def logSoftmax(x):
    #     xdev = x - x.max(axis=0, keepdims=True)
    #     return xdev - T.log(T.sum(T.exp(xdev), axis=0, keepdims=True))
    #
    # def softmax(x):
    #     e_x = T.exp(x - x.max(axis=0, keepdims=True))
    #     return e_x / e_x.sum(axis=0, keepdims=True)

    def back_propagation(net_forward):
        target = T.ivector('target')

        # cost = -T.sum(target * T.log(net_forward), axis=0)

        # cost = T.std(net_forward - target)

        # cost = -T.mean(target * T.log(net_forward) + (1.0 - target) * T.log(1.0 - net_forward))

        cost = -T.sum(target * net_forward, axis=0)

        # cost = -T.sum(target * logSoftmax(net_forward) + (1.0 - target) * T.log(1.0 + 1e-12 - softmax(net_forward)))

        # hinge loss
        # cost = T.max(T.nnet.relu(1 - target * net_forward))

        # print(net_forward.eval())
        # print(cost.eval({target:[-1,-1,-1,-1,1,-1,-1,-1]}))

        # res = target.nonzero()
        # corrects = net_forward[res]
        # rest = theano.tensor.reshape(net_forward[(1 - target).nonzero()],
        #                              (-1, author_amount - 1))
        # rest = theano.tensor.max(rest, axis=1)
        # # just tricky hack with [0] ??!!!??
        # cost = theano.tensor.nnet.relu(rest - corrects + 1)[0]

        if mode == BuildMode.train:
            used_params = list(used_embeddings.values()) + list(parameters.b.values()) + list(parameters.w.values())

            updates = adadelta(cost, used_params)

            return function([target], [cost, T.exp(net_forward)], updates=updates)
        else:
            return function([target], [cost, T.exp(net_forward)])

    f_builder(net[-1])

    net_forward = net[-1].forward

    pydotprint(net_forward, 'net_forward.jpg', format='jpg')

    raise Exception('print ends')

    if mode == BuildMode.train or mode == BuildMode.validation:
        return back_propagation(net_forward)
    else:
        return function([], T.exp(net_forward))
