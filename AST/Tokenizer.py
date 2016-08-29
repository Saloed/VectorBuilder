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


def ast_to_list(ast) -> list:
    nodes = []
    leafs = []
    tokenize(ast, None, None, None, nodes, leafs)
    nodes.extend(leafs)
    reorder(nodes)
    return nodes


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


def build_ast(filename) -> list:
    file = open(filename, 'r')
    code = ""
    lines = file.readlines()
    for line in lines:
        code += line
    ast = parse.parse(code)
    # print_ast(ast)
    # print(print_classes())
    return ast_to_list(ast)


def tokenize(root, parent, parent_id, pos, nodes, leafs):
    if root is None:
        return
    if isinstance(root, Node):
        token_type = root.__repr__()
        token = Token(token_type, token_map[token_type],
                      parent_id, pos)
        if parent is not None:
            parent.children.append(token)
        children = root.children
        if len(children) == 0:
            leafs.append(token)
        else:
            nodes.append(token)

            cur_id = len(nodes)
            for idx, child in enumerate(children):
                tokenize(child, token, cur_id, idx, nodes, leafs)

    elif isinstance(root, list):
        for el in root:
            tokenize(el, parent, parent_id, pos, nodes, leafs)
    else:
        return


def reorder(nodes):
    nodes.reverse()
    length = len(nodes)
    for node in nodes:
        if node.parent is not None:
            node.parent = length - node.parent
