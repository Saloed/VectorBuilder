import theano.tensor as T
from theano import function

from AST.Tokenizer import ast_to_list
from TBCNN.Connection import Connection, PoolConnection
from TBCNN.Layer import *
from TBCNN.NetworkParams import *


def compute_leaf_num(root, nodes, depth=0):
    if len(root.children) == 0:
        root.leaf_num = 1
        root.children_num = 1
        return 1, 1, depth  # leaf_num, children_num
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


def construct_from_ast(ast, parameters: Params, need_back_prop=False):
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

    _, _, avg_depth = compute_leaf_num(nodes[-1], nodes)
    avg_depth *= .6
    if avg_depth < 1:        avg_depth = 1

    leafs = []
    non_leafs = []
    for node in nodes:
        if len(node.children) == 0:
            leafs.append(node)
        else:
            non_leafs.append(node)
    return construct_network(Nodes(nodes, leafs, non_leafs), parameters, need_back_prop, avg_depth)


Nodes = namedtuple('Nodes', ['all_nodes', 'leafs', 'non_leafs'])


def build_net(nodes: Nodes, params: Params, pool_cutoff):
    used_embeddings = {}
    nodes_amount = len(nodes)

    emb_layers = [Layer] * nodes_amount

    for i, node in enumerate(nodes.all_nodes):
        emb = params.embeddings[node.token_index]
        used_embeddings[node.token_index] = emb
        emb_layers[i] = Embedding(emb, "embedding_" + str(i))

    ae_layers = [Layer] * len(nodes.non_leafs)
    cmb_layers = [Layer] * len(nodes.non_leafs)

    for i, node in enumerate(nodes.non_leafs):
        emb_layer = emb_layers[node.pos]
        ae_layers[i] = ae_layer = Encoder(params.b['b_construct'], "autoencoder_" + str(i))
        cmb_layers[i] = cmb_layer = Combination("combination_" + str(i))
        Connection(ae_layer, cmb_layer, params.w['w_comb_ae'])
        Connection(emb_layer, cmb_layer, params.w['w_comb_emb'])
        for child in node.children:
            if child.left_rate != 0:
                Connection(emb_layers[child.pos], ae_layer, params.w['w_left'],
                           w_coeff=child.left_rate * child.leaf_num / node.leaf_num)
            if child.right_rate != 0:
                Connection(emb_layers[child.pos], ae_layer, params.w['w_right'],
                           w_coeff=child.right_rate * child.leaf_num / node.leaf_num)

    pool_top = Pooling('pool_top', NUM_CONVOLUTION)
    pool_left = Pooling('pool_left', NUM_CONVOLUTION)
    pool_right = Pooling('pool_right', NUM_CONVOLUTION)

    conv_layers = []

    queue = [(nodes_amount - 1, None)]
    layer_cnt = 0

    cur_len = len(queue)
    while cur_len != 0:
        next_queue = []
        for (i, info) in queue:

            cur_layer = layers[i]
            cur_node = nodes[i]

            conv_layer = Layer(params.b['b_conv'], "convolve_" + str(i), NUM_CONVOLUTION)
            layers.append(conv_layer)

            Connection(cur_layer, conv_layer, params.w['w_conv_root'])

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
                    Connection(child_layer, conv_layer, params.w['w_conv_left'], left_w)
                if right_w != 0:
                    Connection(child_layer, conv_layer, params.w['w_conv_right'], right_w)

            queue = next_queue

        layer_cnt += 1
        cur_len = len(queue)

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

    layers = emb_layers + ae_layers + cmb_layers + conv_layers

    layers.append(pool_top)
    layers.append(pool_left)
    layers.append(pool_right)

    layers.append(dis_layer)
    layers.append(out_layer)
    return layers


def construct_network(nodes: Nodes, parameters: Params, need_back_prop: bool, pool_cutoff):
    net = build_net(nodes, parameters, pool_cutoff)

    def f_builder(layer):
        if not layer.f_initialized:
            for c in layer.in_connection:
                if not c.f_initialized:
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
