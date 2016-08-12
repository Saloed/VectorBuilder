from AST.Token import Token


def ast_to_list(ast):
    nodes = []
    leafs = []
    tokenize(ast, 'root', None, None, nodes, leafs)
    nodes.extend(leafs)
    reorder(nodes)
    return nodes


def tokenize(root, token_type, parent, pos, nodes, leafs):
    if token_type is None:
        token_type = root.__class__.__name__

    # TODO add token indexes or another idea
    token = Token(token_type, index,
                  parent, pos)

    children = root.children()

    if len(children) == 0:
        leafs.append(token)
    else:
        nodes.append(token)

    cur_id = len(nodes)
    for idx, (name, child) in enumerate(children):
        tokenize(child, None, cur_id, idx, nodes, leafs)


def reorder(nodes):
    nodes.reverse()
    length = len(nodes)
    for node in nodes:
        if node.parent is not None:
            node.parent = length - node.parent
