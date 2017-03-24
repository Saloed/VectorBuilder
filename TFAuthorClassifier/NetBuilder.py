import tensorflow as tf

from TFAuthorClassifier.TFParameters import NUM_CONVOLUTION, BATCH_SIZE, L2_PARAM


class Net:
    def __init__(self, out, loss, error, max_loss, result, target, pc, dropout):
        self.out = out
        self.loss = loss
        self.error = error
        self.max_loss = max_loss
        self.placeholders = pc
        self.dropout = dropout
        self.result = result
        self.target = target


class Placeholders:
    def __init__(self):
        self.node_conv = None
        self.nodes = None
        self.node_children = None
        self.node_emb = None
        self.node_left_c = None
        self.node_right_c = None
        self.length = None
        self.zero_conv_index = None
        self.target = None

    def assign(self, placeholders):
        pc = placeholders  # type: Placeholders
        return {self.length: pc.length,
                self.nodes: pc.nodes,
                self.zero_conv_index: pc.zero_conv_index,
                self.node_emb: pc.node_emb,
                self.node_children: pc.node_children,
                self.node_left_c: pc.node_left_c,
                self.node_right_c: pc.node_right_c,
                self.node_conv: pc.node_conv,
                self.target: pc.target}


def create_convolution(params):
    embeddings = params.embeddings
    pc = Placeholders()
    with tf.name_scope('Placeholders'):
        pc.nodes = tf.placeholder(tf.int32, [None], 'node_indexes')
        pc.node_children = tf.placeholder(tf.int32, [None, None], 'node_children')
        pc.node_emb = tf.placeholder(tf.int32, [None], 'node_emb')
        pc.node_left_c = tf.placeholder(tf.float32, [None], 'left_c')
        pc.node_right_c = tf.placeholder(tf.float32, [None], 'right_c')
        pc.node_conv = tf.placeholder(tf.int32, [None], 'node_conv')
        pc.length = tf.placeholder(tf.int32, (), 'length')
        pc.zero_conv_index = tf.placeholder(tf.int32, (), 'zero_conv_index')

    with tf.name_scope('Target'):
        pc.target = tf.placeholder(tf.int64, [1], 'target')

    def loop_cond(_, i):
        return tf.less(i, pc.length)

    with tf.name_scope('TreeConvolution'):
        convolution = tf.TensorArray(
            dtype=tf.float32,
            size=pc.length + 1,
            dynamic_size=True,
            clear_after_read=False,
            element_shape=[1, NUM_CONVOLUTION]
        )
        convolution.write(pc.zero_conv_index, tf.zeros([1, NUM_CONVOLUTION], tf.float32))

        def convolve_subtree(_convolution, i):
            with tf.name_scope('SubtreeConvolution'):
                node_index = tf.gather(pc.nodes, i)
                children_index = tf.gather(pc.node_children, i)
                root_emb = tf.gather(embeddings, tf.gather(pc.node_emb, node_index))
                children_emb = tf.gather(embeddings, tf.gather(pc.node_emb, children_index))
                children_conv = _convolution.gather(tf.gather(pc.node_conv, children_index))
                children_conv = tf.reshape(children_conv, [-1, NUM_CONVOLUTION])
                node_conv_index = tf.gather(pc.node_conv, node_index)
                children_l_c = tf.gather(pc.node_left_c, children_index)
                children_r_c = tf.gather(pc.node_right_c, children_index)
                children_l_c = tf.expand_dims(children_l_c, 1)
                children_r_c = tf.expand_dims(children_r_c, 1)
                root_emb = tf.expand_dims(root_emb, 0)

                root_conv = tf.matmul(root_emb, params.w['w_conv_root'])

                children_emb = tf.matmul(children_emb, params.w['w_conv_emb'])
                combined = tf.concat([children_emb, children_conv], 0)
                combined = tf.reduce_sum(combined, 0, True)

                def _combine_children(ch_w, ch_coeff):
                    convolved = tf.matmul(combined, ch_w)
                    return tf.multiply(convolved, ch_coeff)

                left_conv = _combine_children(params.w['w_conv_left'], children_l_c)
                right_conv = _combine_children(params.w['w_conv_right'], children_r_c)

                z = tf.concat([left_conv, right_conv, root_conv], 0)
                z = tf.reduce_sum(z, 0)
                z = tf.add(z, params.b['b_conv'])
                conv = tf.nn.relu(z)
                _convolution.write(node_conv_index, conv)
            return _convolution, i + 1

        tf.while_loop(loop_cond, convolve_subtree, [convolution, 0])
        return convolution.read(0), pc
        # def convolve(pool, i):
        #     root_emb = tf.gather(pc.root_nodes, i)
        #     root_ch_i = tf.gather(pc.node_children, i)
        #     children_emb_i = tf.gather(pc.node_emb, root_ch_i)
        #     root = tf.gather(embeddings, root_emb)
        #     root = tf.expand_dims(root, 0)
        #     children_emb = tf.gather(embeddings, children_emb_i)
        #     children_l_c = tf.gather(pc.node_left_c, root_ch_i)
        #     children_r_c = tf.gather(pc.node_right_c, root_ch_i)
        #     children_l_c = tf.expand_dims(children_l_c, 1)
        #     children_r_c = tf.expand_dims(children_r_c, 1)
        #
        #     root = tf.matmul(root, params.w['w_conv_root'])
        #
        #     left_ch = tf.matmul(children_emb, params.w['w_conv_left'])
        #     right_ch = tf.matmul(children_emb, params.w['w_conv_right'])
        #
        #     left_ch = tf.multiply(left_ch, children_l_c)
        #     right_ch = tf.multiply(right_ch, children_r_c)
        #
        #     z = tf.concat([left_ch, right_ch, root], 0)
        #     z = tf.reduce_sum(z, 0)
        #     z = tf.add(z, params.b['b_conv'])
        #     conv = tf.nn.relu(z)
        #
        #     pool = pool.write(i, conv)
        #     i = tf.add(i, 1)
        #     return pool, i
        #
        # with tf.name_scope('Pooling'):
        #     pooling = tf.TensorArray(
        #         tf.float32,
        #         size=0,
        #         dynamic_size=True,
        #         clear_after_read=False,
        #         element_shape=[1, NUM_CONVOLUTION])
        #     pooling, _ = tf.while_loop(loop_cond, convolve, [pooling, 0])
        #     convolution = tf.reduce_max(pooling.concat(), 0, keep_dims=True)

        # return convolution, pc


def create(params):
    placeholders = []
    convolutions = []
    targets = []
    with tf.name_scope('Convolution'):
        for _ in range(BATCH_SIZE):
            conv, pc = create_convolution(params)
            convolutions.append(conv)
            placeholders.append(pc)
            targets.append(pc.target)
        convolution = tf.concat(convolutions, axis=0, name='convolution')
    target = tf.concat(targets, axis=0, name='target')
    with tf.name_scope('Hidden'):
        hid_layer = tf.nn.relu(tf.matmul(convolution, params.w['w_hid']) + params.b['b_hid'])
        dropout_prob = tf.placeholder(tf.float32)
        hid_layer = tf.nn.dropout(hid_layer, dropout_prob)
    with tf.name_scope('Out'):
        logits = tf.matmul(hid_layer, params.w['w_out'] + params.b['b_out'])
        out = tf.nn.softmax(logits)
        result = tf.argmax(out, 1)
    with tf.name_scope('Error'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target)
        error = tf.cast(tf.not_equal(result, target), tf.float32)
        loss = tf.reduce_mean(loss)
        error = tf.reduce_mean(error)
        max_loss = tf.reduce_max(loss)
    return Net(out, loss, error, max_loss, result, target, placeholders, dropout_prob)


def build_net(params):
    net = create(params)
    with tf.name_scope('L2_Loss'):
        reg_weights = [tf.nn.l2_loss(p) for p in params.w.values()]
        l2 = L2_PARAM * tf.reduce_sum(reg_weights)
    cost = net.loss + l2
    updates = tf.train.AdamOptimizer().minimize(cost)
    summaries = tf.summary.merge_all()
    return updates, net, summaries
