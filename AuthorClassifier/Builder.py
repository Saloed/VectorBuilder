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
def convolve_creator(root_node, pooling_layer, conv_layers, layers, params, parameters_amount: dict):
    child_len = len(root_node.children)
    # if child_len == 0: return
    conv_layer = Convolution(params.b['b_conv'], "convolve_" + str(root_node))
    conv_layers.append(conv_layer)
    root_layer = layers[root_node.index]
    Connection(root_layer, conv_layer, params.w['w_conv_root'])
    PoolConnection(conv_layer, pooling_layer)

    parameters_amount['w_conv_root'] += 1
    parameters_amount['b_conv'] += 1

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

        convolve_creator(child, pooling_layer, conv_layers, layers, params, parameters_amount)


def build_net(nodes: Nodes, params: Params, pool_cutoff, authors_amount, parameters_amount: dict):
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

        # parameters_amount['b_construct'] += 1
        # parameters_amount['w_comb_ae'] += 1
        # parameters_amount['w_comb_emb'] += 1

        for child in node.children:
            if child.left_rate != 0:
                # parameters_amount['w_left'] += 1
                Connection(_layers[child.index], ae_layer, params.w['w_left'],
                           w_coeff=child.left_rate * child.leaf_num / node.leaf_num)
            if child.right_rate != 0:
                # parameters_amount['w_right'] += 1
                Connection(_layers[child.index], ae_layer, params.w['w_right'],
                           w_coeff=child.right_rate * child.leaf_num / node.leaf_num)

    # pool_top = Pooling('pool_top', NUM_CONVOLUTION)
    # pool_left = Pooling('pool_left', NUM_CONVOLUTION)
    # pool_right = Pooling('pool_right', NUM_CONVOLUTION)

    # conv_params = ConvolveParams(pool_cutoff, _layers, params, pool_top, pool_left, pool_right)
    conv_layers = []

    # convolve_creator(nodes.root_node, 'unknown', conv_layers, 0, conv_params)

    pooling_layer = Pooling('pool', NUM_CONVOLUTION)
    convolve_creator(nodes.root_node, pooling_layer, conv_layers, _layers, params, parameters_amount)

    # dis_layer = FullConnected(params.b['b_dis'], activation=T.tanh,
    #                           name='discriminative', feature_amount=NUM_DISCRIMINATIVE)
    #
    # def softmax(x):
    #     e_x = T.exp(x - x.max(axis=0, keepdims=True))
    #     return e_x / e_x.sum(axis=0, keepdims=True)

    # def logSoftmax(x):
    #     xdev = x - x.max(axis=0, keepdims=True)
    #     return xdev - T.log(T.sum(T.exp(xdev), axis=0, keepdims=True))

    # out_layer = FullConnected(params.b['b_out'],  # activation=lambda x: x,
    #                           activation=softmax,  # activation=logSoftmax,  # T.nnet.softmax,   # not work (????)
    #                           name="softmax", feature_amount=authors_amount)
    # Connection(pooling_layer, dis_layer, params.w['w_dis_top'])
    #
    # out_layer = RBF_SVM(params.svm['b_out'], params.svm['w_out'],
    #                     # params.svm['c_out'], params.svm['s_out'],
    #                     authors_amount)

    out_layer = FullConnected(params.svm['b_out'], T.tanh, name='out_layer', feature_amount=authors_amount)
    Connection(pooling_layer, out_layer, params.svm['w_out'])

    # because need just pass forward of diss layer to svm
    # PoolConnection(dis_layer, out_layer)

    # PoolConnection(pooling_layer,out_layer)

    # Connection(pool_top, dis_layer, params.w['w_dis_top'])
    # Connection(pool_left, dis_layer, params.w['w_dis_left'])
    # Connection(pool_right, dis_layer, params.w['w_dis_right'])

    layers = _layers + conv_layers

    # layers.append(pool_top)
    # layers.append(pool_left)
    # layers.append(pool_right)

    layers.append(pooling_layer)

    # layers.append(dis_layer)
    layers.append(out_layer)
    return layers, used_embeddings, pooling_layer


# # numerically stable log-softmax with crossentropy
# xdev = x-x.max(1,keepdims=True)
# lsm = xdev - T.log(T.sum(T.exp(xdev),axis=1,keepdims=True))
# sm2 = T.exp(lsm) # just used to show equivalence with sm
# cm2=-T.sum(y*lsm,axis=1)
# g2 = T.grad(cm2.mean(),x)

def construct_network(nodes: Nodes, parameters: Params, mode: BuildMode, pool_cutoff, author_amount):
    parameters_amount = {}
    for k in parameters.w.keys():
        parameters_amount[k] = 0
    for k in parameters.b.keys():
        parameters_amount[k] = 0

    net, used_embeddings, dis_layer = build_net(nodes, parameters, pool_cutoff, author_amount, parameters_amount)

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
        target = T.fvector('target')

        # cost = T.mean(T.nnet.binary_crossentropy(net_forward, target))
        error = T.mean(T.neq(T.round(net_forward), target))

        # cost = -T.sum(target * T.log(net_forward), axis=0)

        # cost = T.std(net_forward - target) + 1e-2

        cost = T.sqrt(T.mean(T.sqr(net_forward - target) + 1e-10))

        # def hinge(self, u):
        #     return T.maximum(0, 1 - u)
        #
        # def ova_svm_cost(self, y1):
        #     """ return the one-vs-all svm cost
        #     given ground-truth y in one-hot {-1, 1} form """
        #     y1_printed = theano.printing.Print('this is important')(T.max(y1))
        #     margin = y1 * self.output
        #     cost = self.hinge(margin).mean(axis=0).sum()
        #     return cost

        # margin = target * net_forward
        # cost = T.maximum(0, 1 - margin).mean(axis=0).sum()
        # cost = -T.mean(target * T.log(net_forward) + (1.0 - target) * T.log(1.0 - net_forward))

        # cost = -T.sum(target * net_forward, axis=0)

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
            # used_embs = list(used_embeddings.values())
            # grads_embs = T.grad(cost, used_embs)

            # used_params = list(parameters.b.values()) + list(parameters.w.values())
            # params_keys = list(parameters.b.keys()) + list(parameters.w.keys())
            params_keys = ['w_conv_left', 'w_conv_right', 'w_conv_root']
            used_params = [parameters.w[k] for k in params_keys]
            params_keys.append('b_conv')
            used_params.append(parameters.b['b_conv'])

            grad_params = T.grad(cost, used_params)

            for i, k in enumerate(params_keys):
                grad_params[i] = grad_params[i] / parameters_amount[k]

            # updates = adadelta(grad_params + grads_embs, used_params + used_embs)
            updates = adadelta(grad_params, used_params)

            svm_params = list(parameters.svm.values())
            svm_updates = adadelta(cost, svm_params)

            # updates = sgd(cost, used_params, 0.0001)
            # svm_updates = sgd(cost, svm_params, 0.0001)

            # updates = adam(cost, used_params)
            # svm_updates = adadelta(cost, svm_params)

            # updates = nesterov_momentum(cost, used_params, 0.1)
            # svm_updates = adam(cost, svm_params)

            return function([target], [cost, error, net_forward], updates=updates
                            # , mode=NanGuardMode(True, True, True, 'None')
                            ), \
                   function([target], [cost, error, net_forward], updates=svm_updates
                            # , mode=NanGuardMode(True, True, True, 'None')
                            )
        else:
            return function([target], [cost, error, net_forward])

    f_builder(net[-1])

    net_forward = net[-1].forward
    tbcnn_out = dis_layer.forward

    # pydotprint(net_forward,'net_fwd.jpg',format='jpg')

    if mode == BuildMode.train or mode == BuildMode.validation:
        return back_propagation(net_forward)
    else:
        return function([], net_forward)
