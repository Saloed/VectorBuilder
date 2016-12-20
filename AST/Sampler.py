import os
import _pickle as pickle

from AST.Token import Token
from AST.Tokenizer import *


class PreparedAST:
    def __init__(self, positive, training_token, training_token_index, ast_len):
        self.positive = positive
        self.training_token = training_token
        self.training_token_index = training_token_index
        self.ast_len = ast_len
        self.eval_set_file = None


class ASTSet:
    def __init__(self, all_tokens, asts):
        self.token_set = all_tokens
        self.ast_set = asts


def build_psi_text(dataset_dir):
    parser = parser_init()
    files = os.listdir(dataset_dir)
    psi_text_file = []
    for file in files:
        try:
            print(file)
            psi_text = get_psi_text(dataset_dir + file, parser)
            psi_text_file.append(psi_text)
        except Exception as ex:
            print(ex)
            continue
    print('end ast building')
    text = ''
    for psi_t in psi_text_file:
        text = text + psi_t + '\n'
    text *= 20
    with open(dataset_dir + '../psi_text.data', 'w') as text_file:
        text_file.write(text)


def build_asts(dataset_dir):
    parser = parser_init()
    files = os.listdir(dataset_dir)
    tokens = get_all_available_tokens(parser)
    data_ast = []
    for file in files:
        try:
            print(file)
            ast = build_ast(dataset_dir + file, parser)
            data_ast.append(ast)
        except Exception as ex:
            print(ex)
            continue
    print('end ast building')
    ast_set = ASTSet(tokens, data_ast)
    with open(dataset_dir + '../ast_file', mode='wb') as ast_file:
        pickle.dump(ast_set, ast_file)


def copy_with_depth(node: Token, depth=1, parent=None) -> Token:
    node_copy = Token(node.token_type, parent, node.is_leaf, node.pos, node.start_line, node.end_line)
    if parent is not None:
        parent.children.append(node_copy)
    if depth != 0:
        for child in node.children:
            copy_with_depth(child, depth - 1, node_copy)
    else:
        node_copy.is_leaf = True
    return node_copy


def prepare_ast(full_ast: Nodes):
    nodes_with_depth = []

    def compute_depth(node: Token, depth):
        if not node.is_leaf:
            nodes_with_depth.append((node, depth))
            for child in node.children:
                compute_depth(child, depth + 1)

    compute_depth(full_ast.root_node, 0)
    nodes_with_depth.sort(key=lambda tup: tup[1], reverse=True)
    return [tree_to_list(copy_with_depth(node[0])) for node in nodes_with_depth]


def generate_samples(data, ast_list: list, training_token_index):
    methods = divide_by_methods(data)
    for m in methods:
        prepared = prepare_ast(m)
        for ast in prepared:
            ast_len = len(ast)
            training_token = ast[training_token_index]

            assert training_token_index == 0

            ast_list.append(PreparedAST(ast, training_token, training_token_index, ast_len))
