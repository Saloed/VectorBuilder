class NetPart:
    def __init__(self, fun, upd, valid):
        self.fun = fun
        self.upd = upd
        self.valid = valid


class NetParts:
    def __init__(self):
        self.classifier = None
        self.pooling = None
        self.conv_join = None
        self.convolution = None
        self.net_function = None
        self.net_validate = None


class ConvUnit:
    def __init__(self, size, root, inputs, conv):
        self.size = size
        self.root = root
        self.inputs = inputs
        self.conv = conv
        self.fun = None
        self.upd = None
        self.valid = None


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


def construct_from_nodes(ast: Nodes, parameters: Params, mode: BuildMode, author_amount, parts):
    # visualize(ast.root_node, 'ast.png')
    nodes = ast.all_nodes
    compute_rates(ast.root_node)
    _, _, avg_depth = compute_leaf_num(ast.root_node, nodes)
    avg_depth *= .6
    if avg_depth < 1:        avg_depth = 1

    return construct_network(ast, parameters, mode, author_amount, parts)


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


def update_function(loss, params, normalize_const):
    grad_params = T.grad(loss, params)
    normalized = [g / normalize_const for g in grad_params]
    return adadelta(normalized, params)


def build_convolve_updates(parts: NetParts, params: Params, authors_amount):
    for size, conv in parts.convolution.items():
        if conv.fun is None:
            conv_layer = conv.conv
            pool = Pooling('pooling')
            classifier = FullConnected(params.svm['b_out'], T.tanh, name='out_layer', feature_amount=authors_amount)
            PoolConnection(conv_layer, pool)
            Connection(pool, classifier, params.svm['w_out'])
            build_net_forward(classifier)
            target = T.fvector('target')
            cost = loss_function(classifier.forward, target)
            if conv.size > 0:
                used_params = [params.w['w_conv_root'], params.w['w_conv_left'], params.w['w_conv_right'],
                               params.b['b_conv']]
            else:
                used_params = [params.w['w_conv_root'], params.b['b_conv']]

            norm_const = T.fscalar('norm_const')
            updates = update_function(cost, used_params, norm_const)
            conv.upd = updates
            arguments = [i.forward for i in conv.inputs]
            arguments.insert(0, conv.root.forward)
            conv.fun = function([target, norm_const] + arguments, conv_layer.forward,
                                updates=updates, name='conv_{}'.format(conv.size))
            conv.valid = function(arguments, conv_layer.forward,
                                  name='conv_valid_{}'.format(conv.size))


def convolve_creator(root_node, pooling_layer, conv_layers, layers, params, parameters_amount: dict, parts):
    child_len = len(root_node.children)
    # if child_len == 0: return
    conv_layer = Convolution(params.b['b_conv'], "convolve_" + str(root_node), size=child_len)
    conv_layers.append(conv_layer)
    root_layer = layers[root_node.index]
    Connection(root_layer, conv_layer, params.w['w_conv_root'])
    PoolConnection(conv_layer, pooling_layer)

    parameters_amount['w_conv_root'] += 1
    parameters_amount['b_conv'] += 1

    if child_len not in parts.convolution:
        inputs = [Placeholder(T.fvector('conv_{}'.format(i)), 'conv_{}'.format(i), NUM_FEATURES) for i in
                  range(child_len)]
        root = Placeholder(T.fvector('conv_root'), 'conv_root', NUM_FEATURES)
        conv_layer = Convolution(params.b['b_conv'], "convolve_{}".format(child_len), size=child_len)
        Connection(root, conv_layer, params.w['w_conv_root'])

        for pos, child in enumerate(inputs):
            if child_len == 1:
                left_w = .5
                right_w = .5
            else:
                right_w = pos / (child_len - 1.0)
                left_w = 1.0 - right_w
            if left_w != 0:
                Connection(child, conv_layer, params.w['w_conv_left'], left_w)
            if right_w != 0:
                Connection(child, conv_layer, params.w['w_conv_right'], right_w)

        parts.convolution[child_len] = ConvUnit(child_len, root, inputs, conv_layer)

    for child in root_node.children:
        if child_len == 1:
            left_w = .5
            right_w = .5
        else:
            right_w = child.pos / (child_len - 1.0)
            left_w = 1.0 - right_w

        child_layer = layers[child.index]
        if left_w != 0:
            parameters_amount['w_conv_left'] += 1
            Connection(child_layer, conv_layer, params.w['w_conv_left'], left_w)
        if right_w != 0:
            parameters_amount['w_conv_right'] += 1
            Connection(child_layer, conv_layer, params.w['w_conv_right'], right_w)

        convolve_creator(child, pooling_layer, conv_layers, layers, params, parameters_amount, parts)


def build_net(nodes: Nodes, params: Params, authors_amount, parameters_amount: dict, parts: NetParts):
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

        parameters_amount['b_construct'] += 1
        parameters_amount['w_comb_ae'] += 1
        parameters_amount['w_comb_emb'] += 1

        for child in node.children:
            if child.left_rate != 0:
                parameters_amount['w_left'] += 1
                Connection(_layers[child.index], ae_layer, params.w['w_left'],
                           w_coeff=child.left_rate * child.leaf_num / node.leaf_num)
            if child.right_rate != 0:
                parameters_amount['w_right'] += 1
                Connection(_layers[child.index], ae_layer, params.w['w_right'],
                           w_coeff=child.right_rate * child.leaf_num / node.leaf_num)

    conv_layers = []

    pooling_layer = Pooling('pool', NUM_CONVOLUTION)
    convolve_creator(nodes.root_node, pooling_layer, conv_layers, _layers, params, parameters_amount, parts)

    out_layer = FullConnected(params.svm['b_out'], T.tanh, name='out_layer', feature_amount=authors_amount)
    Connection(pooling_layer, out_layer, params.svm['w_out'])

    layers = _layers + conv_layers

    layers.append(pooling_layer)

    layers.append(out_layer)
    return used_embeddings, _layers, out_layer


def loss_function(net_forward, target):
    return T.sqrt(T.mean(T.sqr(net_forward - target) + 1e-10))


def error_function(net_forward, target):
    return (T.neq(T.round(net_forward), target)).mean()


def build_parts(params: Params, authors_amount) -> NetParts:
    parts = NetParts()
    target = T.fvector('tar')

    cl_in = T.fvector('cl_in')
    cl_input = Placeholder(cl_in, 'classifier_in', NUM_CONVOLUTION)
    classifier = FullConnected(params.svm['b_out'], T.tanh, 'out_layer', authors_amount)
    Connection(cl_input, classifier, params.svm['w_out'])
    cl_params = [params.svm['b_out'], params.svm['w_out']]
    build_net_forward(classifier)
    cl_frwd = classifier.forward
    cl_loss = loss_function(cl_frwd, target)
    cl_error = error_function(cl_frwd, target)
    cl_update = adadelta(cl_loss, cl_params)
    cl_fun = function([cl_in, target], [cl_frwd, cl_loss, cl_error], updates=cl_update, name='classifier')
    cl_valid = function([cl_in, target], [cl_frwd, cl_loss, cl_error], name='classifier_valid')
    parts.classifier = NetPart(cl_fun, cl_update, cl_valid)

    pool_in = T.fmatrix('pool_in')
    pool_frwd = T.max(pool_in, axis=0)
    pool_fun = function([pool_in], pool_frwd, name='pooling')
    parts.pooling = NetPart(pool_fun, [], pool_fun)

    def net_eval_function(nodes: Nodes, embeddings: list, target):
        return parts.classifier.fun(parts.pooling.fun([
                                                          parts.convolution[len(node.children)].fun(
                                                              *([target, len(node.children) + 1,
                                                                 embeddings[node.index]] + [
                                                                    embeddings[c.index]
                                                                    for c in
                                                                    node.children]))
                                                          for node in nodes.all_nodes]), target)

    def net_valid_functoin(nodes: Nodes, embeddings: list, target):
        return parts.classifier.valid(parts.pooling.valid([
                                                              parts.convolution[len(node.children)].valid(
                                                                  *([embeddings[node.index]] + [
                                                                      embeddings[c.index]
                                                                      for c in
                                                                      node.children]))
                                                              for node in nodes.all_nodes]), target)

    parts.net_function = net_eval_function
    parts.net_validate = net_valid_functoin
    parts.convolution = {}

    return parts


def build_net_forward(layer: Layer):
    if layer.forward is None:
        for c in layer.in_connection:
            if c.forward is None:
                build_net_forward(c.from_layer)
                c.build_forward()
        layer.build_forward()


def back_propagation(out, net_forward, parameters, parameters_amount, used_embeddings):
    target = T.fvector('target')
    # cost = loss_function(net_forward, target)
    # used_embs = list(used_embeddings.values())
    # grads_embs = T.grad(cost, used_embs)
    #
    # params_keys = ['w_left', 'w_right', 'w_comb_ae', 'w_comb_emb']
    # used_params = [parameters.w[k] for k in params_keys]
    # params_keys.append('b_construct')
    # used_params.append(parameters.b['b_construct'])
    #
    # grad_params = T.grad(cost, used_params)
    #
    # for i, k in enumerate(params_keys):
    #     grad_params[i] = grad_params[i] / parameters_amount[k]
    #
    # updates = update_function(grad_params + grads_embs, used_params + used_embs)

    return function([target], [c.forward for c in out], on_unused_input='ignore')  # , updates=updates)


def construct_network(nodes: Nodes, parameters: Params, mode: BuildMode, author_amount, parts):
    parameters_amount = {}
    for k in parameters.w.keys():
        parameters_amount[k] = 0
    for k in parameters.b.keys():
        parameters_amount[k] = 0

    used_embeddings, layers, out = build_net(nodes, parameters, author_amount, parameters_amount, parts)
    build_convolve_updates(parts, parameters, author_amount)
    build_net_forward(out)
    if mode == BuildMode.train:
        return back_propagation(layers, out.forward, parameters, parameters_amount, used_embeddings)
    else:
        return function([], [c.forward for c in layers])
