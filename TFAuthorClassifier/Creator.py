import _pickle as P
import logging
import sys
from collections import OrderedDict, namedtuple
from random import shuffle

import numpy as np
import tensorflow as tf

from AST.Token import Token
from AST.Tokenizer import Nodes
from TFAuthorClassifier.TFParameters import NUM_FEATURES, Params, RANDOM_RANGE, NUM_CONVOLUTION, NUM_HIDDEN, BATCH_SIZE, \
    l2_param, NUM_EPOCH, SAVE_PERIOD
from Utils.Visualization import new_figure, update_figure, save_to_file
from Utils.Wrappers import timing


class Net:
    def __init__(self, out, loss, error, max_loss, pc):
        self.out = out
        self.loss = loss
        self.error = error
        self.max_loss = max_loss
        self.placeholders = pc


class Placeholders:
    def __init__(self, root_nodes, node_children, node_emb, node_left_coef, node_right_coef, target):
        self.root_nodes = root_nodes
        self.node_children = node_children
        self.node_emb = node_emb
        self.node_left_coef = node_left_coef
        self.node_right_coef = node_right_coef
        self.target = target

    def assign(self, placeholders):
        pc = placeholders  # type: Placeholders
        return {self.root_nodes: pc.root_nodes,
                self.node_emb: pc.node_emb,
                self.node_children: pc.node_children,
                self.node_left_coef: pc.node_left_coef,
                self.node_right_coef: pc.node_right_coef,
                self.target: pc.target}


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


def prepare_batch(ast: Nodes, emb_indexes, r_index):
    target = [r_index[ast.root_node.author]]
    nodes = ast.all_nodes
    compute_rates(ast.root_node)
    compute_leaf_num(ast.root_node, nodes)
    ast.non_leafs.sort(key=lambda x: x.index)
    ast.all_nodes.sort(key=lambda x: x.index)
    root_nodes = [emb_indexes[node.token_type] for node in ast.non_leafs]
    node_emb = [emb_indexes[node.token_type] for node in ast.all_nodes]
    node_left_coef = [node.left_rate for node in ast.all_nodes]
    node_right_coef = [node.right_rate for node in ast.all_nodes]
    zero_node_index = len(node_emb)
    node_emb.append(emb_indexes['ZERO_EMB'])
    node_left_coef.append(0.0)
    node_right_coef.append(0.0)
    max_children_len = max([len(node.children) for node in ast.non_leafs])

    def align_nodes(nodes):
        result = [node.index for node in nodes]
        while len(result) != max_children_len:
            result.append(zero_node_index)
        return result

    node_children = [align_nodes(node.children) for node in ast.non_leafs]

    return Placeholders(root_nodes, node_children, node_emb, node_left_coef, node_right_coef, target)


def rand_weight(shape_0, shape_1, name):
    with tf.name_scope(name):
        var = tf.Variable(
            tf.truncated_normal(shape=[shape_1, shape_0], stddev=RANDOM_RANGE),
            name=name)
        variable_summaries(var)
    return var


def rand_bias(shape, name):
    return rand_weight(shape, 1, name)


def init_params(author_amount):
    with open('TFAuthorClassifier/embeddings', 'rb') as f:
        np_embs = P.load(f)
    with tf.name_scope('Embeddings'):
        np_embs = OrderedDict(np_embs)
        zero_emb = np.zeros([NUM_FEATURES], np.float32)
        np_embs['ZERO_EMB'] = zero_emb
        emb_indexes = {name: i for i, name in enumerate(np_embs.keys())}
        embeddings = tf.stack(list(np_embs.values()))

    with tf.name_scope('Params'):
        w_conv_root = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_root')
        w_conv_left = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_left')
        w_conv_right = rand_weight(NUM_CONVOLUTION, NUM_FEATURES, 'w_conv_right')
        w_hid = rand_weight(NUM_HIDDEN, NUM_CONVOLUTION, 'w_hid')
        w_out = rand_weight(author_amount, NUM_HIDDEN, 'w_out')

        b_conv = rand_bias(NUM_CONVOLUTION, 'b_conv')
        b_hid = rand_bias(NUM_HIDDEN, 'b_hid')
        b_out = rand_bias(author_amount, 'b_out')

    return Params(w_conv_root, w_conv_left, w_conv_right,
                  w_hid, w_out, b_conv, b_hid, b_out,
                  embeddings), emb_indexes


def create_convolution(params):
    embeddings = params.embeddings
    root_nodes = tf.placeholder(tf.int32, [None], 'node_indexes')
    node_children = tf.placeholder(tf.int32, [None, None], 'node_children')
    node_emb = tf.placeholder(tf.int32, [None], 'node_emb')
    node_left_coef = tf.placeholder(tf.float32, [None], 'left_coef')
    node_right_coef = tf.placeholder(tf.float32, [None], 'right_coef')
    target = tf.placeholder(tf.int64, [1], 'target')

    pooling = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        element_shape=[1, NUM_CONVOLUTION])

    def loop_cond(_, i):
        return tf.less(i, tf.squeeze(tf.shape(root_nodes)))

    def convolve(pool, i):
        root_emb = tf.gather(root_nodes, i)
        root_ch_i = tf.gather(node_children, i)
        children_emb_i = tf.gather(node_emb, root_ch_i)
        root = tf.gather(embeddings, root_emb)
        root = tf.expand_dims(root, 0)
        children_emb = tf.gather(embeddings, children_emb_i)
        children_l_coef = tf.gather(node_left_coef, root_ch_i)
        children_r_coef = tf.gather(node_right_coef, root_ch_i)
        children_l_coef = tf.expand_dims(children_l_coef, 1)
        children_r_coef = tf.expand_dims(children_r_coef, 1)

        root = tf.matmul(root, params.w['w_conv_root'])

        left_ch = tf.matmul(children_emb, params.w['w_conv_left'])
        right_ch = tf.matmul(children_emb, params.w['w_conv_right'])

        left_ch = tf.multiply(left_ch, children_l_coef)
        right_ch = tf.multiply(right_ch, children_r_coef)

        z = tf.concat([left_ch, right_ch, root], 0)
        z = tf.reduce_sum(z, 0)
        z = tf.add(z, params.b['b_conv'])
        conv = tf.nn.relu(z)

        pool = pool.write(i, conv)
        i = tf.add(i, 1)
        return pool, i

    with tf.name_scope('Convolution'):
        pooling, _ = tf.while_loop(loop_cond, convolve, [pooling, 0])
        convolution = tf.reduce_max(pooling.concat(), 0, keep_dims=True)
    return convolution, Placeholders(root_nodes, node_children, node_emb, node_left_coef, node_right_coef, target)


def create(params):
    placeholders = []
    convolutions = []
    targets = []
    for _ in range(BATCH_SIZE):
        conv, pc = create_convolution(params)
        convolutions.append(conv)
        placeholders.append(pc)
        targets.append(pc.target)
    convolution = tf.stack(convolutions, name='convolution')
    target = tf.stack(targets, name='target')
    with tf.name_scope('Hidden'):
        hid_layer = tf.nn.sigmoid(tf.matmul(convolution, params.w['w_hid']) + params.b['b_hid'])
    with tf.name_scope('Out'):
        logits = tf.matmul(hid_layer, params.w['w_out'] + params.b['b_out'])
        out = tf.nn.softmax(logits)
    with tf.name_scope('Error'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target)
        error = tf.cast(tf.not_equal(tf.argmax(out, 1), target), tf.float32)
        loss = tf.reduce_mean(loss)
        error = tf.reduce_mean(error)
        max_loss = tf.reduce_max(loss)
    return Net(out, loss, error, max_loss, placeholders)


def divide_data_set(data_set, train_units, valid_units, test_units):
    data_set = list(zip(list(data_set.values())))
    if train_units + valid_units + test_units > len(data_set):
        raise Exception('Too much units to divide')
    shuffle(data_set)
    train_set = [d for data in data_set[:train_units] for d in data]
    v_end = train_units + valid_units
    valid_set = [d for data in data_set[train_units:v_end] for d in data]
    test_set = [d for data in data_set[v_end:v_end + test_units] for d in data]
    return train_set, valid_set, test_set


def build_vectors(authors):
    size = len(authors)
    index = {author: uauthor[0] for uauthor in authors for author in uauthor[1]}
    return size, index


def generate_batches(dataset, emb_indexes, r_index, net):
    size = len(dataset) // BATCH_SIZE
    pc = net.placeholders
    batches = []
    for j in range(size):
        ind = j * BATCH_SIZE
        d = dataset[ind:ind + BATCH_SIZE]
        feed = {}
        for i in range(BATCH_SIZE):
            feed.update(pc[i].assign(prepare_batch(d[i], emb_indexes, r_index)))
        batches.append(feed)
    return batches


def build_net(params):
    net = create(params)
    with tf.name_scope('L2_Loss'):
        reg_weights = [tf.nn.l2_loss(p) for p in params.w.values()]
        l2 = l2_param * tf.reduce_sum(reg_weights)
    cost = net.loss + l2
    updates = tf.train.AdamOptimizer().minimize(cost)
    summaries = tf.summary.merge_all()
    return updates, net, summaries


@timing
def process_set(batches, fun, is_train, session):
    res = ([], [], [])
    for feed in batches:
        if is_train:
            loss, max_loss, err, _ = session.run(fetches=fun, feed_dict=feed)
        else:
            loss, max_loss, err = session.run(fetches=fun, feed_dict=feed)
        res[0].append(float(loss))
        res[1].append(float(max_loss))
        res[2].append(float(err))
    loss = np.mean(res[0])
    loss_max = np.max(res[1])
    err = np.mean(res[2])
    return loss, loss_max, err


DataSet = namedtuple('DataSet', ['test', 'valid', 'train', 'r_index', 'amount'])


def divide_dataset():
    with open('Dataset/CombinedProjects/top_authors_MPS', 'rb') as f:
        # with open('TFAuthorClassifier/test_data', 'rb') as f:
        dataset = P.load(f)
    dataset = dataset[:5]
    indexes = range(len(dataset))
    authors = [(i, dataset[i][1]) for i in indexes]
    authors_amount, r_index = build_vectors(authors)
    batches = {i: dataset[i][0] for i in indexes}
    train_set, valid_set, test_set = divide_data_set(batches, 100, 50, 100)
    dataset = DataSet(test_set, valid_set, train_set, r_index, authors_amount)
    with open('Dataset/CombinedProjects/top_authors_MPS_data', 'wb') as f:
        P.dump(dataset, f)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main():
    logging.basicConfig(filename='log.txt', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    with open('Dataset/CombinedProjects/top_authors_MPS_data', 'rb') as f:
        dataset = P.load(f)  # type: DataSet
    params, emb_indexes = init_params(dataset.amount)
    updates, net, summaries = build_net(params)
    train_set = generate_batches(dataset.train, emb_indexes, dataset.r_index, net)
    test_set = generate_batches(dataset.valid, emb_indexes, dataset.r_index, net)
    saver = tf.train.Saver()
    for retry_num in range(5):
        plot_axes, plot = new_figure(retry_num, NUM_EPOCH, 2)
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess, tf.device('/cpu:0'):
            summary_writer = tf.summary.FileWriter('Summary', sess.graph)
            sess.run(tf.global_variables_initializer())
            for train_epoch in range(NUM_EPOCH):
                shuffle(train_set)
                tr_loss, tr_max, tr_err = process_set(train_set, [net.loss, net.max_loss, net.error, updates], True,
                                                      sess)
                te_loss, te_max, te_err = process_set(test_set, [net.loss, net.max_loss, net.error], False, sess)

                print_str = [
                    'epoch {0} retry {1}'.format(train_epoch, retry_num),
                    'train | mean {0:.4f} | max {1:.4f} | percent {2:.2f}'.format(float(tr_loss),
                                                                                  float(tr_max),
                                                                                  float(tr_err)),
                    'test  | mean {0:.4f} | max {1:.4f} | percent {2:.2f}'.format(float(te_loss),
                                                                                  float(te_max),
                                                                                  float(te_err)),
                    '################'
                ]
                logging.info('\n'.join(print_str))
                if train_epoch % SAVE_PERIOD == 0:
                    saver.save(sess, 'TFAuthorClassifier/NewParams/model', retry_num * 10000 + train_epoch)
                update_figure(plot, plot_axes, train_epoch, te_loss, tr_loss)
                info = sess.run(fetches=[summaries])
                summary_writer.add_summary(info, train_epoch)
        save_to_file(plot, 'retry{}.png'.format(retry_num))
        summary_writer.close()


if __name__ == '__main__':
    main()
