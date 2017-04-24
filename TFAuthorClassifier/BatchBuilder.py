from AST.Structures import Token, Nodes
from TFAuthorClassifier.NetBuilder import Placeholders
from TFAuthorClassifier.TFParameters import BATCH_SIZE


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


def compute_indexes(root_node: Token):
    def _indexing_tree(_root_node, start_index, criteria):
        index = start_index
        queue = [_root_node]
        while len(queue) > 0:
            node = queue[0]
            del queue[0]
            for child in node.children:
                if criteria(child):
                    index += 1
                    child.index = index
            for child in node.children:
                queue.append(child)
        return index

    root_node.index = 0
    new_index = _indexing_tree(root_node, 0, lambda node: not node.is_leaf)
    _indexing_tree(root_node, new_index, lambda node: node.is_leaf)


def _make_list(fun, iterable):
    size = len(iterable)
    l = [None] * size
    for itm in iterable:
        l[itm.index] = fun(itm)
    return l


def prepare_batch(ast: Nodes, emb_indexes, r_index):
    pc = Placeholders()
    pc.target = [r_index[ast.root_node.author]]
    compute_indexes(ast.root_node)
    compute_rates(ast.root_node)
    ast.non_leafs.sort(key=lambda x: x.index, reverse=True)
    ast.all_nodes.sort(key=lambda x: x.index, reverse=True)

    zero_token = Token('ZERO_EMB', None, True)
    zero_token.index = len(ast.all_nodes)
    zero_token.left_rate = 0.0
    zero_token.right_rate = 0.0
    ast.all_nodes.append(zero_token)

    pc.node_emb = _make_list(lambda itm: emb_indexes[itm.token_type], ast.all_nodes)
    pc.node_left_c = _make_list(lambda itm: itm.left_rate, ast.all_nodes)
    pc.node_right_c = _make_list(lambda itm: itm.right_rate, ast.all_nodes)

    max_children_len = max([len(node.children) for node in ast.non_leafs])

    def align_nodes(_nodes):
        result = [node.index for node in _nodes]
        while len(result) != max_children_len:
            result.append(zero_token.index)
        return result

    pc.node_children = [align_nodes(node.children) for node in ast.non_leafs]
    pc.nodes = [node.index for node in ast.non_leafs]
    pc.zero_conv_index = len(ast.non_leafs)
    pc.node_conv = _make_list(lambda node: pc.zero_conv_index if node.is_leaf else node.index, ast.all_nodes)
    pc.length = len(ast.non_leafs)
    return pc


def generate_batches(data_set, emb_indexes, r_index, net, dropout):
    size = len(data_set) // BATCH_SIZE
    pc = net.placeholders
    batches = []
    for j in range(size):
        ind = j * BATCH_SIZE
        d = data_set[ind:ind + BATCH_SIZE]
        feed = {net.dropout: dropout}
        for i in range(BATCH_SIZE):
            feed.update(pc[i].assign(prepare_batch(d[i], emb_indexes, r_index)))
        batches.append(feed)
    return batches
