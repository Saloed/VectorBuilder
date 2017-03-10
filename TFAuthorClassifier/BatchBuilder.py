from AST.Token import Token
from AST.Tokenizer import Nodes
from TFAuthorClassifier.NetBuilder import Placeholders
from TFAuthorClassifier.TFParameters import BATCH_SIZE


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
    pc = Placeholders()
    pc.target = [r_index[ast.root_node.author]]
    nodes = ast.all_nodes
    compute_rates(ast.root_node)
    compute_leaf_num(ast.root_node, nodes)
    ast.non_leafs.sort(key=lambda x: x.index)
    ast.all_nodes.sort(key=lambda x: x.index)
    pc.root_nodes = [emb_indexes[node.token_type] for node in ast.non_leafs]
    pc.node_emb = [emb_indexes[node.token_type] for node in ast.all_nodes]
    pc.node_left_coef = [node.left_rate for node in ast.all_nodes]
    pc.node_right_coef = [node.right_rate for node in ast.all_nodes]
    zero_node_index = len(pc.node_emb)
    pc.node_emb.append(emb_indexes['ZERO_EMB'])
    pc.node_left_coef.append(0.0)
    pc.node_right_coef.append(0.0)
    max_children_len = max([len(node.children) for node in ast.non_leafs])

    def align_nodes(nodes):
        result = [node.index for node in nodes]
        while len(result) != max_children_len:
            result.append(zero_node_index)
        return result

    pc.node_children = [align_nodes(node.children) for node in ast.non_leafs]

    return pc


def generate_batches(data_set, emb_indexes, r_index, net):
    size = len(data_set) // BATCH_SIZE
    pc = net.placeholders
    batches = []
    for j in range(size):
        ind = j * BATCH_SIZE
        d = data_set[ind:ind + BATCH_SIZE]
        feed = {}
        for i in range(BATCH_SIZE):
            feed.update(pc[i].assign(prepare_batch(d[i], emb_indexes, r_index)))
        batches.append(feed)
    return batches
