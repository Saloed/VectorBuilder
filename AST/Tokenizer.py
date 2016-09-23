from collections import namedtuple

from AST.Token import Token
from AST.TokenMap import token_map
from javalang.parser import parse
from javalang.ast import *
from javalang import *
import javalang.tree as T
import sys, inspect

# def print_classes():
#     tt = dict()
#     counter = 0
#     for name, obj in inspect.getmembers(sys.modules[T.__name__]):
#         if inspect.isclass(obj):
#             temp = obj  # .__init__(temp)
#             tt[temp.__name__] = counter
#             counter += 1
#     return tt

Nodes = namedtuple('Nodes', ['root_node', 'all_nodes', 'non_leafs'])


def ast_to_nodes(ast) -> list:
    all_nodes = []
    tokenize(ast, None, None, 0, all_nodes)
    root_node = all_nodes[0]
    non_leafs = []
    for node in all_nodes:
        if len(node.children) != 0:
            non_leafs.append(node)
    reorder(all_nodes)
    return Nodes(root_node, all_nodes, non_leafs)


def print_ast(ast, shift=""):
    if ast is None:
        return
    if isinstance(ast, Node):
        print(shift, ast)
        for child in ast.children:
            print_ast(child, shift + '\t')
    elif isinstance(ast, list):
        for el in ast:
            print_ast(el, shift)
    else:
        return


def print_tokens(tokens):
    def print_token(token, shift=""):
        print(shift, token.token_type)
        for child in token.children:
            print_token(child, shift + "\t")

    print_token(tokens[0])


def build_ast(filename) -> list:
    file = open(filename, 'r')
    code = ""
    lines = file.readlines()
    for line in lines:
        code += line
    ast = parse.parse(code)
    # print_ast(ast)
    # print(print_classes())
    return ast_to_nodes(ast)


def tokenize(root, parent, parent_id, pos, nodes):
    if root is None:
        return 0
    if isinstance(root, Node):
        token_type = root.__repr__()
        token = Token(token_type, token_map[token_type],
                      parent, pos)
        nodes.append(token)
        if parent is not None:
            parent.children.append(token)

        children = root.children
        cur_id = len(nodes)
        child_num = 0
        for child in children:
            child_num += tokenize(child, token, cur_id, child_num, nodes)
        return 1
    elif isinstance(root, list):
        child_num = pos
        for el in root:
            child_num += tokenize(el, parent, parent_id, child_num, nodes)
        return child_num - pos
    else:
        return 0


def reorder(nodes):
    nodes.reverse()
    for i, node in enumerate(nodes):
        node._hash = i
