import subprocess
import sys
import time
from copy import deepcopy

from py4j.java_gateway import GatewayParameters, JavaGateway, get_field

from AST.Structures import Token, Author, Nodes


def _indexate(nodes) -> list:
    for i, node in enumerate(nodes):
        node.index = i
    return nodes


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


def analyzer_init(port=None):
    if port is None:
        port = 25537
    process = subprocess.Popen(
        ["java", "-jar", "AST/AuthorAnalyzer.jar", str(port)],
        stdout=sys.stdout, stderr=sys.stderr)
    time.sleep(1)
    parameters = GatewayParameters(port=port)
    gateway = JavaGateway(gateway_parameters=parameters)
    analyzer = gateway.entry_point.getMain()
    return analyzer, gateway, process


def _parse_tree(root, parent=None, pos=0) -> Token:
    node = _parse_token(root, parent, pos)
    if parent is not None:
        parent.children.append(node)
    children = get_field(root, 'children')
    for i, child in enumerate(children):
        _parse_tree(child, node, i)
    return node


def author_collapse(authors):
    acopy = deepcopy(authors)
    uauthors = []

    while len(acopy) > 0:
        change = True
        tmp = [acopy[0]]
        del acopy[0]
        while change:
            change = False
            for i in range(len(acopy)):
                if acopy[i] in tmp:
                    tmp.append(acopy[i])
                    acopy[i] = None
                    change = True
            acopy = [a for a in acopy if a is not None]
        uauthors.append(tmp)
    return uauthors


def process_ast(ast):
    root_node = _parse_tree(ast)
    author = get_field(ast, 'author')
    if author is not None:
        author = Author(get_field(author, 'name'), get_field(author, 'email'))
        root_node.author = author
    all_nodes = tree_to_list(root_node)
    non_leafs = [node for node in all_nodes if not node.is_leaf]
    return Nodes(root_node, all_nodes, non_leafs)
