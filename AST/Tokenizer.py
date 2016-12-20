from collections import namedtuple
import pydot
from py4j.java_gateway import JavaGateway, get_field

from AST.Token import Token

Nodes = namedtuple('Nodes', ['root_node', 'all_nodes', 'non_leafs'])


def parser_init():
    gateway = JavaGateway()
    parser = gateway.entry_point.getParser()
    return parser


def tree_to_list(tree_root_token: Token) -> list:
    def convert(token: Token, tokens: list):
        tokens.append(token)
        for child in token.children:
            convert(child, tokens)

    token_list = []
    convert(tree_root_token, token_list)
    return _indexate(token_list)


def _parse_token(node, parent, pos) -> Token:
    node_name = get_field(node, 'nodeName')
    is_terminal = get_field(node, 'isTerminal')
    src_start = get_field(node, 'sourceStart')
    src_end = get_field(node, 'sourceEnd')
    return Token(node_name, parent, is_terminal, pos, src_start, src_end)


def _parse_tree(root, parent=None, pos=0) -> Token:
    node = _parse_token(root, parent, pos)
    if parent is not None:
        parent.children.append(node)
    children = get_field(root, 'children')
    for i, child in enumerate(children):
        _parse_tree(child, node, i)
    return node


def _indexate(nodes) -> list:
    for i, node in enumerate(nodes):
        node.index = i
    return nodes


def get_psi_text(filename, parser):
    return str(parser.parsePSIText(filename))


def build_ast(filename, parser):
    ast = parser.parseFile(filename)
    root_node = _parse_tree(ast)
    all_nodes = tree_to_list(root_node)
    non_leafs = [node for node in all_nodes if not node.is_leaf]
    return Nodes(root_node, all_nodes, non_leafs)


def _shifted_string(token: Token, shift="") -> str:
    string = '\n{}┗ {} [{}:{}]'.format(shift, str(token), token.start_line, token.end_line)
    for child in token.children:
        if child.is_leaf:
            string += " " + str(child)
        else:
            string += _shifted_string(child, shift + " ")
    return string


def get_all_available_tokens(parser) -> list:
    tokens = parser.getAllAvailableTokens()
    return [str(token) for token in tokens]


def _method_search(root: Token, methods: list) -> list:
    if root.token_type == 'METHOD':
        root.parent.children.remove(root)
        root.parent = None
        methods.append(root)
    else:
        for child in root.children:
            _method_search(child, methods)
    return methods


def divide_by_methods(ast: Nodes) -> list:
    methods = _method_search(ast.root_node, [])
    if len(methods) == 0:
        return []
    else:
        def _to_nodes(m) -> Nodes:
            root_node = m
            all_nodes = tree_to_list(root_node)
            non_leafs = [node for node in all_nodes if not node.is_leaf]
            return Nodes(root_node, all_nodes, non_leafs)

        return [_to_nodes(m) for m in methods]


def print_ast(ast_root_node):
    print(_shifted_string(ast_root_node))


def _create_graph(node, graph):
    node_name = str(node)
    for child in node.children:
        graph.add_edge(pydot.Edge(node_name, str(child)))
        _create_graph(child, graph)


def visualize(ast_root_node, filename):
    graph = pydot.Dot(graph_type='graph')
    _create_graph(ast_root_node, graph)
    graph.write_png(filename)


if __name__ == '__main__':
    parser = parser_init()
    tokens = get_all_available_tokens(parser)
    print(tokens)
    test_filename = '../Dataset/java_files/LayoutEngine.java'
    ast = build_ast(test_filename, parser)
    print_ast(ast.root_node)
